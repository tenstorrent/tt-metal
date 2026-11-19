# Pending Ornith integration and main-native rebuild

Everything in this directory is preparation only. No main source/native output was changed, no native build executed, and no device was opened. Apply only after untouched32 comparison and explicit64 device validation pass, with the serving device/process reserved by the parent.

`ornith_sampling64.patch` targets exactly three main paths under `models/autoports/ornith_ai_ornith_1_5_9b/tt/`: `generator.py`, `sampling_state.py`, `generator_vllm.py`. `before/` and `after/` preserve exact source copies; `main_source_sha256.json` pins the original snapshot. `git apply --check --whitespace=error` passed against current main, and all three main hashes remained unchanged at handoff.

The generator defines `DEVICE_SAMPLING_MAX_TOP_K = 64`, uses it in constructor, validation and every formatter/raw-parameter admission path, and the adapter advertises literal64 for readiness parser compatibility. Adapter initialization rejects any mismatch with the imported canonical generator constant before model creation. The slot-state helper derives candidate capacity from `self.sampler.tt_sampling.max_top_k` for scoped prior/update/upload and remap. Its standalone `_raw_parameters` keyword defaults32 for existing callers. Physical `CAPACITY`, sampler batch, lane validation, seed/history buffers and slot mappings stay32. Raw temperatures remain raw until the common uploader formats them once. Stochastic k65, zero and unrestricted-negative k are rejected by generator admission; greedy preserves its established k1 sentinel behavior.

Eight CPU-only tests passed in 0.058s (`host_tests.log`): actual formatter/raw function, generator validation, actual prefill admission statements, actual scoped reset and remap methods, complete call-site propagation, constructor/adapter capacity and physical32 consistency, literal capability parsing and mismatch rejection before model initialization. The test extracts production AST from scratch files and imports no TTNN; prefill's surrounding model/device work is deliberately excluded. Run:

```bash
I=/home/ttuser/ornith-9b/tt-metal-sampling64/sampling64_validation/ornith_integration
/home/ttuser/ornith-9b/tt-metal/python_env/bin/python "$I/test_ornith_capacity_host.py"
```

Two existing main device rejection fixtures still use k64 and must become k65 when integration is approved: `tests/full_model_prefill_generation_probe.py:52` and `tests/full_model_serving_decode_probe.py:169`. They were not edited here. Serving contract artifacts recording the historical32 boundary remain historical evidence; new support needs a separate validated result.

## Main-native rebuild plan (not executed)

A normal `ninja -C build_Release -n ttnn/_ttnn.so` currently reports CMake regeneration and an old build-log version. Therefore a minimal ordinary target cannot presently be guaranteed to avoid unrelated rebuilding. `rebuild_main_native.py` instead records/replays exactly the existing commands for three affected full unity objects, then `_ttnncpp.so` and `_ttnn.so`, all using main source/output paths. It removes PCH inclusion to avoid stale operation signatures; dependency-file and object outputs remain their normal main locations. Commands execute serially with180s timeout each, and both links force `--threads=1 --thinlto-jobs=1`. No CMake/Ninja build is run. Exact current argv and cwd are already recorded in `main_rebuild_plan.json`.

Default invocation is plan-only and has been run. The following **future parent-only** invocation writes normal main outputs and must wait until source integration and all queued device validation are approved:

```bash
MAIN=/home/ttuser/ornith-9b/tt-metal
V=/home/ttuser/ornith-9b/tt-metal-sampling64/sampling64_validation
I="$V/ornith_integration"
# Parent applies the validated native patch, common Python patch and model patch first.
# git -C "$MAIN" apply "$V/sampling64.patch"
# git -C "$MAIN" apply "$V/python_plumbing.patch"
# git -C "$MAIN" apply "$I/ornith_sampling64.patch"
"$MAIN/python_env/bin/python" "$I/rebuild_main_native.py" --execute
```

The script refuses compilation unless main native sources match `native_expected_sha256.json`. It backs up currently installed native libraries, records source hashes, full commands, elapsed times and returncodes, links in normal `build_Release/ttnn/`, then atomically copies the two successful outputs to the existing served locations `build_Release/lib/_ttnncpp.so` and `ttnn/ttnn/_ttnn.so`. No private-library path is added. It performs a final import-only provenance assertion; two loaded `_ttnn` native libraries and the Python package must all resolve under main, and the new keyword must be visible. Results/backups go in timestamped `main_rebuild_*/` beneath this directory. If a compile/link fails, deployment does not occur; main objects may be partially rebuilt and logs identify the step. Compilation/link duration was about94s in isolated validation; main duration is unmeasured.

For subsequent serving, explicitly use the normal main environment (never inherit the isolated library path):

```bash
unset LD_PRELOAD
export TT_METAL_HOME=/home/ttuser/ornith-9b/tt-metal
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib"
# Keep the normal serving TT_METAL_CACHE setting; do not use validation/device_cache.
"$TT_METAL_HOME/python_env/bin/python" "$I/check_main_import.py"
```

The import check opens no device. Restart serving after deployment rather than using an already-loaded process. The native/common/model changes still need integrated TP4 sampling, trace/state and actual P2 measurements; these CPU checks establish no device support or performance claim.
