# Prefill migration device fixtures

These are standalone prefill-side tests. The receiver is passive allocated KV memory; it loads no decoder model. The production model, cache writer, acknowledgment ordering and native manager are unchanged. The source origin table records which copied helpers are byte-identical to the tested fixtures and which files change configuration/import paths or repository-required formatting.

## Test surfaces

| Directory | Device workload | Expected coverage |
|---|---|---|
| `writer_boundaries/` | Production cache writer and independent tensor/table oracle; no weights, H2D runtime or manager | One test, 128 seed writes, 10 boundary writes, 65,536-page before and after snapshots, 208 touched pages, 5,104 valid and 1,552 padding rows. |
| `runtime_edges/` | One full32 source owner, real persistent H2D, no native manager | Five calls, 160 post-sync acknowledgments, six full-cache snapshots; two slots, partial tails, [32,65) continuation and runtime-only slot reuse. |
| `native_ranges/` | Two retained owners, real H2D source and native managers, passive destination | Six generations, seven full32 calls, 224 post-sync acknowledgments, ordinary/crossed mappings, selected prefix/continuation/reuse, exact selected and untouched packed pages. |
| `native_cancel/` | Two retained owners, real H2D/model source and native cancellation/restart | Two scenario calls, 64 real acknowledgments, delayed destination, retained allocation restart and 512 exact packed pages. |
| `native_capacity/` | Real full32 source and crossed selected native tails at a4K allocation | Two prompts4,096/4,064;8 real calls,256 acknowledgments,32,768 exact pages /136MiB; untouched samples and manager memory observations. |

The writer and runtime fixtures passed on silicon before this packaging change. The frozen2K native, paired-range and real cancellation/restart fixtures passed their recorded device scopes; the focused4K selected-range gate also passed. Do not infer a new device pass from the host suite or from source equivalence. Existing 2K numerical validation remains the accuracy anchor; these structural/exact-byte tests do not add an HF golden. Valid token endpoints are distinct from copied whole32-token pages. The cancellation fixture has a separately accepted frozen device result; see the [cancellation report](../../docs/migration-prefill-cancel-restart.md). See the [range report](../../docs/migration-prefill-ranges.md) and [focused4K report](../../docs/migration-prefill-capacity-4k.md). The relocated capacity package has host checks, not a new device run. The user subsequently approved larger lengths. Device validation at 8K–64K is resuming and remains pending per size; 128K remains deferred.

## Host checks (no Torch or device imports)

The repository pytest entry launches each suite in a separate, native-import-blocked
Python process. The standalone `checks_*.py` files are excluded from ordinary pytest
discovery, so their local imports and blockers cannot affect later device tests.
From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --noconftest \
  models/demos/llama_3p1_8b_d_p/tests/migration/test_host_contracts.py
```

For a single suite, use the same import-free child entry directly:

```bash
python3 -I -S -B models/demos/llama_3p1_8b_d_p/tests/migration/test_host_contracts.py \
  --suite runtime_edges
```

The five child inventories are17 writer,37 runtime,57 paired-range,46 cancellation and34 capacity cases (191 total). The wrapper also checks actual writer importlib collection.
The real writer device test remains a normal `test_*.py` pytest entry.

The additional publication cases load the actual controller/supervisor/owner imports. They reject wrong manifest hashes, binary hashes, case inventories, library-linkage receipts and assigned host/job/lock before native work. Terminal journal, stale-page and cleanup fault cases remain in the copied suites. Temporary host files use the caller's TMPDIR.

## Configure one reviewed run

All examples are unarmed. They are input schemas, not ready-to-run device permissions. Copy the relevant `plan.example.json` and `model-spec.example.json` outside the checkout. Replace every `/configure` value; the programs intentionally do not substitute shell variables inside JSON. Supply absolute paths on storage visible to both endpoints.

- `model-spec.json`: set `prepared_source` to this source-matched tt-metal checkout. Retain the exact checkpoint metadata hashes and two-slot/32-layer/1024-chunk geometry. The original scenarios use2K; the closed capacity example selects4K through its own validated plan. Bind `checkpoint` to the matching Llama-3.1-8B-Instruct weights.
- `environment_script`: configure a copy of `environment.example.sh`, including the built source-matched Python/TTNN and native loader paths. The node script verifies this file's plan-bound SHA before sourcing it. The included `verify_native_env.py` checks actual imports/libraries and does not open a mesh. Node-owned source/library observations still run later.
- Bind a fresh run directory, nonce, Slurm owner, exact assignment, lease, per-node physical lock, ports and independently accepted recent health receipts. Lock names include rack and node, for example `prefill-device-120-c03u14.lock`. The node and job must agree with the real environment. Source and passive are distinct devices. Existing timing/resource limits and retained-owner recovery remain unchanged.
- Bind absolute source/binary/library paths and their SHA256s in `source_pins` (runtime) or `pins` (paired). Include every local non-test Python helper, shell entry point, scenario/fixture/model spec, production model/cache/layout/export dependencies, environment script, probe, health receipt, native binaries and loaded libraries. For paired runs, `accepted_source_pins` is a separate JSON path-to-hash map of the source-matched production files; pin that map too. The controller records the entire configured map before, on-node and after execution.
- For paired runs, supply a current source-manifest file for the reviewed tt-d-gen source/passive bridge and its SHA. Supply the real host-validation result using `bridge-host-validation.example.json`'s schema:48 exact cases, actual/verified0, host-test no-Metal/UMD linkage, and exact source/passive executable hashes. The example does not assert a pass. Keep the actual binary identity distinct from the historical first2K result. `startup_evidence` can bind the published `docs/migration-native-2k.json` as historical native-startup/transport provenance; it does not replace fresh health or the actual same-run seats/library/table/lifecycle observations.
- Do not rewrite failed evidence or copy an old run nonce. Review the final resolved plan and source map, then set the existing review/dispatch flags. Hash the exact plan bytes after review. The examples cannot launch while closed.

For example, create source-map entries with `hashlib.sha256(Path(path).read_bytes()).hexdigest()` using the explicit file list above. Do not store hashes for private preparation copies when the public checkout is the actual imported code. No helper automatically manufactures acceptance or arms a plan.

## Exact device commands

Use only a user-assigned Galaxy, current accepted health, fresh lease and no overlapping node owner. The site scheduler/physical-lock checks remain mandatory. One CPU and the four thread limits above apply. Commands below are interfaces for an already reviewed binding; they are not an allocation or reset recipe.

### Writer boundary singleton

The fixed pytest fixture declares `(4,8)` and `FABRIC_1D_RING`. Its `config.example.json` records the binding checklist and expected counts. Source the configured environment, create a fresh output directory, collect on the assigned node, then run the exact collected case under its exclusive physical lock:

```bash
export LLAMA_PREFILL_EVIDENCE_DIR=/absolute/new-writer-run
"$PREFILL_PYTHON" -m pytest --collect-only -q \
  models/demos/llama_3p1_8b_d_p/tests/migration/writer_boundaries/test_writer_boundaries_device.py
"$PREFILL_PYTHON" -m pytest -q -s \
  'models/demos/llama_3p1_8b_d_p/tests/migration/writer_boundaries/test_writer_boundaries_device.py::test_llama_bfp8_writer_boundaries_preserve_packed_cache[ring-galaxy-4x8]' \
  --junitxml="$LLAMA_PREFILL_EVIDENCE_DIR/pytest.xml"
```

Use the actual collected parameter suffix if pytest renders fixture IDs differently. Require one case, the full `writer-boundaries-report.json` inventory, actual0, exact source/native provenance and clean32-chip close. Existing site/controller ownership guards surround pytest; do not add a forced process-group timeout to a live mesh.

### Runtime owner

```bash
export RUNTIME_EDGE_DISPATCH_AUTHORIZED=root-reviewed
python3 -B models/demos/llama_3p1_8b_d_p/tests/migration/runtime_edges/controller.py \
  --plan /absolute/runtime-plan.json --plan-sha256 EXACT_PLAN_SHA256
```

The controller uses the existing node-owned supervisor. Its child interface is `run_owner.py --plan PLAN --plan-sha256 SHA`; do not call it directly to bypass inherited flock or scheduler evidence. `verify_report.py REPORT --plan OWNER_PLAN` rechecks the160-ack/snapshot report. Require owner cleanup plus actual32-chip close before lock release.

### Paired selected ranges

```bash
python3 -B models/demos/llama_3p1_8b_d_p/tests/migration/native_ranges/controller.py \
  --plan /absolute/range-plan.json --plan-sha256 EXACT_PLAN_SHA256
```

The existing role interface remains `supervise_owner.py --plan PLAN --plan-sha256 SHA --role source` (or `passive`). The controller runs one node step per role; each supervisor retains its own lock and cache owner. `verify_ranges.py --plan PLAN` checks saved reports and bytes. The wrapper waits for each actual native exit and exact clean Cluster lifecycle, then both-manager stop proof, before either cache is released. On ambiguity it records a recovery hold. Do not kill the retained owner or release its lock merely because the other PID vanished.

### Paired cancellation and restart

See [native_cancel/README.md](native_cancel/README.md) for the exact two-epoch scenario, configured dependency closure and closed-plan command. The controller preserves the same plan/hash and role/supervisor interfaces. Native managers must stop on both endpoints in both epochs before either retained cache is released.

### Focused4K capacity reproduction

See [native_capacity/README.md](native_capacity/README.md) for the hash-validated token manifest, pre-native warmup, resource bounds and existing controller/supervisor command. The example is closed at4K. The user has approved resuming 8K–64K validation. Each device run still requires a separately reviewed binding and acceptance receipt.

## Evidence and publication status

See `docs/migration-prefill-address-evidence.md`, `docs/migration-native-2k.md`, and the existing migration test guide for scope. This directory supplies executable tests and commands that were previously private. The packaging checks are host-only. Record source commit, configured file map, actual native executable hashes, exact command/nonce, per-phase reports, exit statuses, cleanup and lease/lock handback for every later device run.
