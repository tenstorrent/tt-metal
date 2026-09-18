# Native cancellation and restart fixture

This optional fixture exercises one real first-chunk cancellation and one real
32-token restart at a 2,048-token cache capacity. It uses the existing prefill
model/runtime, H2D service, native source/passive bridge, and manager. It does not
run a decoder, a larger-context comparison, or a performance benchmark.

## Scenario and acceptance

The source performs the two existing full32 warmup calls before creating the
bridge. Epoch A registers a 2K request, produces its first 1K chunk through H2D
and the real 32-layer model, and emits 32 real layer acknowledgements. The
passive initially holds its expectation: all intended first-chunk sentinel
pages must remain unchanged before the named arm barrier. Cancellation then
requires the exact native terminal records, slot pins released, and both
managers/bridges stopped.

Epoch B retains the original TTNN owner/cache, rewrites passive sentinels,
creates fresh discovery/manager/bridge identities, and produces a distinct
32-token prompt with another 32 real layer acknowledgements. Exact selected
packed source/destination bytes cover all 16 K/V configurations and all 32
layers. The source capture precedes acknowledgement publication and differs
from its pre-request cache. Both epochs must prove native I/O stopped on both
endpoints before cache release; a missing proof retains the owner for recovery.

The checks do not establish cancellation while bytes are in flight, interruption
of a device kernel, abrupt peer-loss recovery, model numerical accuracy, or a
capacity above 2K. The reduced serialized fixtures establish the wire schema:
the passive snapshot has no source-only aggregate success field. They preserve
all fields used by the terminal checker and original capture hashes in
fixtures/provenance.json. They are host regression inputs, not a passing device
receipt. Device acceptance must be recorded separately for the actual attempt.

The [accepted frozen device result](../../../docs/migration-prefill-cancel-restart.md) records the actual cancellation/restart evidence. This portable configuration/path successor remains separately host-validated.

## Configure a single attempt

Use the common migration model-spec.example.json, environment.example.sh and
verify_native_env.py, plus native_ranges/frozen_runtime_owner.py for the
unchanged cache seed helper. Copy plan.example.json into a new attempt
directory, and bind:

- The current two-node assignment, exact allocation owner, role-specific
  physical locks, UTC lease ends, fresh per-node clean health receipts and
  task-local network ports.
- The real model/source/checkpoint inventory, token fixture paths, environment
  script, manager/DMK/libraries, compatible bridge manifest and executable
  hashes. The bridge receipt must contain exactly 48 passing C++ cases,
  no Metal/UMD linkage, and those executable hashes.
- Every local executable helper and its imported source closure in the pin map,
  including the sibling node-run.sh and configured external dependencies.
  The full source/checkpoint inventory remains external; this package contains
  no weights, native libraries, binaries or raw device logs.
- The original scope/provenance receipts and a fresh run nonce. Keep run_dir
  equal to the new attempt directory's run child. Open reviewed and
  launch_authorized only after the concrete assignment/resources are reviewed.

The example remains closed. The existing per-role supervisor bounds are 1,800
seconds normal work, 420 seconds cancellation, and 1,800 seconds recovery reserve.
Controller admission additionally reserves 120 seconds. These are bounds, not
a completion estimate; preflight requires a fresh lease and no active steps.
Each role narrows its process affinity to one CPU. Every run has its own empty
JIT directory. This publication does not change reuse, timeout, native stop or
allocation lifetime policy.

From a repository checkout with the configured plan hash:

    python3 models/demos/llama_3p1_8b_d_p/tests/migration/native_cancel/launch-wrapper.py --plan /absolute/attempt/plan.json --plan-sha256 PLAN_SHA256

The wrapper invokes the sibling controller. Its node entry retains the exact
supervisor/owner interface: --plan, --plan-sha256 and --role source|passive.
The environment script is hash-checked before sourcing. No path in the example
identifies an active machine or deployable private installation.

## Host validation

From the repository root:

    python3 -I -S -B models/demos/llama_3p1_8b_d_p/tests/migration/native_cancel/run_host_checks.py

There are 46 standard-library checks: 42 preserved scenario/lifetime/schema
checks and four portability tests covering the actual owner/supervisor import
routes, manifest/binary/48-case gates, changed helper bytes, environment pinning,
configured ownership and physical lock identity. The runtime ordering tests
extract the actual repository runtime class without importing Torch or TTNN.

The common migration pytest wrapper runs this suite in an isolated child using
-I -S -B. Standalone modules use checks_ names, so ordinary importlib discovery
does not load their local helpers or native-import blocker. The child blocks
model/native imports; parent pytest collection is unaffected.
