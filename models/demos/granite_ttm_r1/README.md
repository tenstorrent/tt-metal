# Granite TTM-R1

## Platforms
Wormhole

## Status
Initial bring-up scaffold for `ibm-granite/granite-timeseries-ttm-r1`.

Current contents:
- HuggingFace reference loader with `trust_remote_code=True`
- reference preprocessing and regression metrics
- TTNN model skeleton with explicit torch fallback path for unported modules
- demo entrypoint using synthetic `context_length=512` and `forecast_length=96`

## Layout
- `models/demos/granite_ttm_r1/reference/`: reference model loading, preprocessing, metrics
- `models/demos/granite_ttm_r1/ttnn/`: TTNN module shells and fallback bridge
- `models/demos/granite_ttm_r1/demo/demo.py`: bring-up entrypoint
- `models/demos/wormhole/granite_ttm_r1/`: Wormhole-specific tests will live here

## Next steps
- inspect the HuggingFace module tree and forward signature
- map real Granite TTM submodules to TTNN module boundaries
- replace torch fallback blocks with native TTNN ops
- add PCC and perf tests once hardware is attached
