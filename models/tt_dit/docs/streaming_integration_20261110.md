# Reviewed H3 source integration

Target: `ltx-rt` at `522ef26916bdd78e2446fd1ff19a4f33e40ecfb6`.
This integration imports only the missing authored streaming modules, their
tests and preparation documents. It does not import the main-based H3 tree.

| Input | Accepted head | Independent review |
|---|---|---|
| Controller | `3f8a885f5f48b087e937747d4994ee6069d6bc07` | #15 |
| Encoder and transport | `14fa5e4d5c50b24402e12c6d616eabcfb21a0439` | #37, run155 |
| Corrected source/config preparation | `f72e0a7be72fd99d0b46407b49802d826981d60f` | #44, run171 |

The four runtime modules are byte-identical to the reviewed transport head.
The two preparation documents are historical snapshots, also imported unchanged;
their queued-review wording describes the earlier inspection. Reviews #37/#44
are now accepted. `STREAMING.md` records current compatibility and qualification.
The run174 authored-delta inventory records each input path and SHA256.

The target base ref2va keyword arguments, reference fields, output metadata and
12/3 scheduler shifts match the bridge. New source-contract tests inspect the
actual target interfaces without importing ttnn. Runtime tensor/codec tests
remain required on an authorized CPU host. No generated-video equivalence or
device execution follows from static compatibility.

The target does not contain `pipeline_minimax_h3_turbo.py` or
`experimental/lora/h3_adapter_loader.py`. Thus its base pipeline cannot provide
the Turbo4/HyperFlow8 speed routes. The preserved wrapper rejects an unadapted
pipeline on either route. Importing those runtimes requires separate scoped
work; this integration does not substitute the base route for `h3-fast`.
The preparation manifest retains the exact, distinct FL2V/R2V Turbo adapters
and public `h3-fast` selector. The wrapper itself only supports ref2va.

LTX implementation remains with task18, and server instrumentation
`c19b5fb3f146f6169e7604785b2263e16f8696f3` remains in ltx-server.
Existing owner request `836a136e3305d732` for candidate
`173bce4455988341ce4358c566d57ec1f9de8a80` and its complete build log had no matching
reply at inspection. That candidate is absent from this local object store;
its claimed unity failure was not reproduced or repaired here. Its conv3d
repair is untouched. Owner557 retains F4 staging and sulphur-mac owns rollout.
Benchmark server `6653391` and identity snapshot `874684a` remain distinct.

Complete event accounting and temporal-prefix conditioning are absent; the
capped receipt-bridge continuation is not implemented. All four qualification
cells remain open under the concurrent caller-throughput contract in the
preparation manifest, including mandatory per-chip TYO2 160 W readback.
Combined-source checks/review, publication and sulphur adoption must each have
their own receipt. Task20 may be marked superseded only after actual re-landing.

[authored by AI]
