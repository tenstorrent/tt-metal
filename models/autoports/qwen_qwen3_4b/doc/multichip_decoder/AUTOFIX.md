# Autofix Note: Watcher and Fabric Router

## Symptom

The multichip watcher stress passes its pytest assertions, but full watcher
coverage with active Ethernet watcher enabled fails outside the Qwen decoder
path.

Commands and outcomes:

| Command variant | Result |
| --- | --- |
| `TT_METAL_WATCHER=10` | Fabric initialization fails with active Ethernet program size above kernel config buffer. |
| `TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1` | Single-mesh stress passes, then watcher aborts on active Ethernet fabric-router NOC packet-tag state. |
| `TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1` | Stress passes, then active Ethernet teardown times out. |
| `TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1` | Stress passes and process exits cleanly. |

## Diagnosis

The stage-owned decoder enters fabric through `ttnn.all_reduce` in
`MultichipDecoder._all_reduce_hidden`. The failing watcher logs identify active
Ethernet fabric-router kernels, not Qwen model kernels.

The failing sanitizer message is emitted by Metal's debug helper for invalid NOC
packet-tag state before the next kernel starts. The active Ethernet kernel named
in the watcher log is `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`.

The source inspection found that fabric-router paths use transaction-ID NOC
writes and wait for acknowledgements/barriers, while a nearby fabric mux path
explicitly clears packet tags after closing. That makes the remaining failure a
fabric-router watcher/firmware cleanup issue, not a multichip decoder tensor
contract issue.

## Stage Workaround

The strongest passing watcher evidence for this stage keeps watcher enabled for
worker-side model execution and disables only Ethernet-core watcher coverage:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN3_4B_MULTICHIP_RUN_WATCHER_STRESS=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_watcher_single_mesh_stress --tb=short
```

Result: `1 passed in 73.51s`, clean process exit.

## Saved Logs

- `watcher/watcher_failed_active_eth_program_size.log`
- `watcher/watcher_failed_repeated_fabric_reinit.log`
- `watcher/watcher_failed_eth_noc_sanitize_after_stress.log`
- `watcher/watcher_failed_noc_sanitize_disabled_eth_teardown.log`
- `watcher/watcher_pass_eth_disabled_single_mesh_stress.log`

## Remaining Risk

This is not full active-Ethernet watcher-clean evidence. It is the strongest
stage-owned evidence available without changing global fabric router firmware or
watcher behavior. The failed full-Ethernet watcher runs passed model-level PCC
checks before failing in fabric-router watcher/teardown paths.
