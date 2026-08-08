# Replacing the `agentic_port` pipeline with a GitHub Agentic Workflow

Notes branch. Nothing here is wired into CI — this directory contains findings and reference
material only, published so the work survives the machine it was done on.

## What this is

`tt-dm-codegen`'s `codegen_agentic_port` branch carries ~31.8k lines under `agentic_port/` that
port `generic_op` implementations into tt-metal as C++ program factories, verify they beat the
native ttnn path, and route regressions back to native. The question under investigation: can that
be replaced by a gh-aw workflow running in tt-metal, on a card, in far fewer lines?

**Short answer: yes.** The one thing that could have killed it — whether an agent can reach a
Tenstorrent card from inside gh-aw's sandbox — was resolved empirically, in our favour.

## Read this

- **[`NOTES.md`](NOTES.md)** — the full writeup. Self-contained. Sections worth jumping to:
  - §9 — the finding that unblocks the design (`mcp-scripts` reach the card; verified)
  - §10 — threat model and hardening rules
  - §11 — accepted risk posture, and which of §10 to actually implement
  - §1–§4 — what the existing pipeline does and where its lines go
- **[`probe/`](probe/) — the experiment that produced §9.** A verified-working example of an
  `mcp-scripts` tool that opens `/dev/tenstorrent` from a gh-aw workflow on an N150. This is the
  pattern the real workflow should be built on, so it is kept as a reference implementation.

## The finding in one table

Three legs, one N150, one job
([run 31195804593](https://github.com/tenstorrent/tt-metal/actions/runs/31195804593)):

| Leg | Hostname | `AWF_*` vars | `open("/dev/tenstorrent/0")` |
|---|---|---|---|
| Ordinary workflow step | `tt-metal-ci-vm-136` | — | OK |
| Agent's own bash, inside AWF | `37f3f355a679` | 11 | **EPERM** |
| `probe_device` mcp-script tool | `tt-metal-ci-vm-136` | 0 | **OK** |

The agent itself cannot touch the card, and no frontmatter escape from that is available — strict
mode is mandatory on a public repo and rejects every one. But `mcp-scripts` run as an HTTP server on
the *runner host*, outside the agent container, reached over `host.docker.internal`. So the agent
calls a declared tool that holds the card while remaining sandboxed itself.

That preserves a tight edit → build → measure → edit loop in a single on-card job, without
weakening the sandbox, and it lands the trust boundary exactly where gh-aw intends. It also mirrors
what the old pipeline already did with `tt-device-mcp` / `DaemonClient`, except the framework
supplies the seam instead of it being hand-built.

## About `probe/`

Recovered from the throwaway branch (commit `2c92359b6a8`) before it was deleted.

These files are **inert here**. `awf-device-probe.md` is a gh-aw workflow source, but it lives
outside `.github/workflows/`, so GitHub will not schedule it; its `push` trigger also names a branch
that no longer exists. To re-run it you would have to deliberately copy it back into
`.github/workflows/` and recompile with `gh aw compile`.

Two caveats if you reuse it as a template, both learned the hard way and both in §9:

- Declare a `noop` safe output. Without one the agent feels obliged to make some safe-output call
  and files junk issues — it did exactly that twice, producing #52480 and #52477 (both closed).
- Tool `timeout` is per call and defaults to 60s. 300 is verified working. A cold `build_metal.sh`
  will exceed anything sensible, so do the first build in a `steps:` pre-step and let the tool do
  incremental rebuilds only.

## Status

Feasibility work is complete; no blocking unknown remains. Next step is a prototype workflow, per
§11's "net effect on the prototype".
