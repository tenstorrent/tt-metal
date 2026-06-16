# RFC: multi-tenant device arbitration for tt-device-mcp

**Target:** `tenstorrent/tt-device-mcp` (baseline `c759849`)
**Status:** draft / for discussion
**Author:** (smarton, via Claude Code) — evidence gathered on shared bh Galaxy `g03blx02`

## Problem

`tt-device-mcp` is a single-host device **arbiter** but is deployed **one daemon per user**
on a hardcoded TCP port, with no cross-user serialization. On a shared 32-chip Galaxy this
breaks concretely (all observed on `g03blx02` this session):

- **Port race.** `cli.py` daemon-start binds a fixed `--port` (default 8333) with no
  fallback and refuses if taken. Second user's daemon loses the bind race; their jobs run
  out-of-band → **two daemons, two queues, one device** → no serialization.
- **Client mis-routing.** The tt-buddy plugin hardcodes `http://localhost:8333/mcp` for
  every user, so a 2nd user's client hits the *1st* user's daemon (whose jobs run as the
  wrong uid → workspace perms fail). Result: users abandon the MCP and run **bare-metal
  pytest**, escaping all coordination → sysmem collisions / wedged mesh.
- **Unscoped reset.** `tt_device_reset` runs `tt-smi -r` with no check for other tenants.
  A reset is a board-level reset of **all 32 chips**: any concurrent run loses the device
  mid-op → aborts or goes **D-state (unkillable) until another reset**. Verified directly:
  `kill -9` on a device-holding process left it D-state + wedged the mesh.
- **Spoofable identity.** `owner` is a self-reported request field (`owner=data.get(...)`);
  `kill`'s owner-check is therefore unauthenticated — any client can claim another's owner.
- **`/tmp` is not shared.** Sessions here have per-session `/tmp` (PrivateTmp); a `/tmp`
  lock serializes nothing cross-user. `/home` is the only shared mount.
- **IDE launch-env fragility.** `${VAR}`-templated client URLs silently fall back to the
  default when the IDE launches Claude without the env var.
- **Jobs exec as the daemon's uid** (`create_subprocess_shell`, no setuid) — so a *single
  shared* daemon can't write other users' workspaces (the perms wall).

## Goals

- One canonical arbiter per host; all tenants serialized on the physical device.
- Per-user execution (correct workspace, perms, accountability) — never "everyone as one uid".
- Bare-metal device use is **prevented**, not punished.
- Zero per-user/per-session manual config; zero runtime privilege for users.
- Reset is safe by construction: never resets while another uid holds the device.

## Non-goals

- Cross-host scheduling. Per-host only.
- Replacing the MCP tool surface (`job_run/_run_bg/status/logs/wait/kill/queue_status/exec/reset` stay).

## Proposed architecture

**Root `tt-device-broker` system service + per-user stdio shim.**

- **Broker (root systemd service, one per host).** Owns: the queue + job registry +
  device serialization + reset-gate + identity. Listens on a **unix domain socket**
  (`/run/tt-device-broker.sock`, world-connectable) — no TCP port, no collisions, and
  `SO_PEERCRED` gives the **real uid** for free (kills spoofable `owner`). The broker does
  **not** run user jobs as root.
- **Per-user stdio shim (MCP server).** Each user's Claude spawns it via the plugin
  (`type:"stdio"`, command-launched — Claude manages lifecycle, no port, no daemon to
  bootstrap). The shim exposes the existing MCP tools, forwards to the broker over the
  socket, and **execs the job as the invoking user in their workspace**.
- **Device gating via cgroup (airtight bare-metal prevention).** cgroup v2 here has **no
  `devices` controller** — gating is done with systemd `DeviceAllow=` (eBPF cgroup-device)
  + a udev rule. A udev rule sets `/dev/tenstorrent/* 0660` (drop today's world `0666`), and
  the broker launches each job in a slice granted the device, e.g.
  `systemd-run --uid=<user> --slice=ttdev --property=DeviceAllow=/dev/tenstorrent/* rw <cmd>`.
  Processes outside that slice **physically cannot open the device** → bare-metal is blocked
  at the kernel: no auto-kill, no race, no wedge.

### Reset gate

Allow reset **iff no `/dev/tenstorrent/*` holder has a uid ≠ caller** (your own wedged
holders are fine — the "reset my own wedge" case). The broker (privileged) enumerates
device-fd holders + uids; cross-references the registry for a friendly name. `force=true`
overrides but **logs + names the foreign holder** it nukes. (Queued jobs are irrelevant —
they aren't on the device yet.)

### Identity / authz

`owner` derived from socket peer uid, never the request body. `kill`/`exec`/`logs` are
owner-scoped by real uid (can't touch another user's job; push-through `exec` only against
your own held job).

## Bootstrap

A **one-time install script run by a sudoer** (`install-tt-device-broker.sh`): drops the
udev rule, installs + `enable --now` the systemd unit (boot-persistent, alongside the
existing `tenstorrent-hugepages.service`), creates the socket dir. **No ssh, no per-user
sudo, no runtime privilege.** tt-buddy at runtime: detect the socket → connect; if absent,
**error to the admin** ("run `sudo install-tt-device-broker.sh`") — it never starts a root
service or ssh's as another user.

## Phased plan

1. **Interim (shipped as out-of-tree hotfix, fixes live installs now):** keep the HTTP
   daemon, add a host `flock` on the shared `/home` mount around every device op, per-UID
   port, user-scoped client registration, a tt-buddy bare-metal guard hook. Stopgap only.
2. **Phase A (in-repo, low risk):** unix-socket transport + `SO_PEERCRED` auth + reset-gate
   (reject on foreign-uid device holder) on the existing daemon. Removes spoofing + the
   worst footgun without the cgroup work.
3. **Phase B (in-repo, the end state):** root broker + udev 0660 + `systemd-run` slice
   admission + per-user stdio shim; bare-metal physically blocked; install script + tt-buddy
   detect-or-error.

## Rejected alternatives

- **Auto-kill bare-metal processes.** Needs root to signal other uids; SIGKILL mid-device-op
  **wedges the mesh** (D-state + forced reset) — the enforcement *causes* the catastrophe;
  reactive (collision already happened); races kill the daemon's own children / legit
  tooling. Replaced by cgroup admission (prevention).
- **Pure stdio + flock only (no daemon).** Loses the shared queue, cross-agent visibility,
  background jobs that outlive a session, centralized job lifecycle, push-through exec, and
  the CLI. Keeps only mutual exclusion. The shim→broker design keeps all of it.
- **Fork the repo.** Two arbiters on one host = the multi-server collision this RFC fixes,
  institutionalized. Must stay one canonical project.
- **`${VAR}` client URL / `/tmp` lock / per-UID port.** Fragile: IDE launch-env drops the
  var; `/tmp` is per-session; per-UID ports still need discovery. Unix socket + peer-uid
  obviates all three.

## Open questions

- systemd-run as root launching arbitrary user commands — injection surface; lock down arg
  handling / use a constrained exec API.
- Privilege the broker needs to enumerate cross-user `/dev/tenstorrent` holders
  (`CAP_DAC_READ_SEARCH` / `CAP_SYS_PTRACE` vs full root).
- Socket-dir perms + cleanup on crash; broker restart with live jobs (re-adopt vs drain).
