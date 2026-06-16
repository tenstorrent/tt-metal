# Phase A: unix-socket transport + SO_PEERCRED authz + reset gate

Implements **Phase A** of the multi-tenant arbitration RFC. Removes the two worst
multi-tenant footguns on a shared host without the root/cgroup work (deferred to
Phase B): spoofable identity, and unscoped device reset.

## What changes

- **Unix-domain-socket transport** alongside the existing HTTP/TCP transport. The
  same ASGI app (`/mcp` + `/api/*` + `/health`) is served over both. Opt-in via
  `--socket <path>` (daemon `start`/`start-fg`) or `TT_DEVICE_MCP_SOCKET`. HTTP is
  unchanged and always on, so existing clients keep working.
- **`SO_PEERCRED` identity.** On the socket, the kernel-provided peer uid is the
  authoritative owner. `kill`/`exec` authz and the reset gate use it; the
  self-reported `owner` request field is ignored for authz on the socket (it
  remains the fallback over HTTP, which has no peer identity — legacy behavior).
- **Reset gate.** `reset_device` refuses a board-level reset while any
  **foreign-uid** process holds `/dev/tenstorrent` — resetting all chips would
  abort another tenant's run mid-op and can wedge the mesh. Resetting over your
  own (possibly wedged) holders is allowed. `force=true` overrides and logs the
  foreign holders it is about to disrupt. Enforced only when the caller's uid is
  known (i.e. over the socket); HTTP keeps legacy unscoped behavior.

## Why

On a shared 32-chip Galaxy, `owner` was a self-reported field (spoofable) and
`tt_device_reset` ran `tt-smi -r` with no tenant check — a reset wedged everyone.
Peer credentials make identity unspoofable; the gate makes reset safe by
construction.

## Tests

37 new unit/integration tests (`test_peercred`, `test_device_holders`,
`test_reset_gate`, `test_authz`, `test_socket_transport` incl. a live socket
JSON-RPC round-trip). All pass; the 62 pre-existing server/cli tests still pass.

## Not in this PR

Phase B (root broker, udev `0660`, `systemd-run` slice admission, per-user stdio
shim, install script — physical bare-metal prevention) lands separately; it needs
root + on-device validation and would make this change un-reviewable if bundled.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
