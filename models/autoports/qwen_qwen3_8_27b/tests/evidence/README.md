# Host-only test evidence

`preparation_events_host_check.json` preserves the synthetic AST-extracted
`prepare_decode` check originally stored under `readiness_vllm/`. It uses fake
tensor and trace operations, a one-layer fixture and a 256-token test context.
It is not a server run, model-context setting, device measurement or model-quality
result. See `../../doc/vllm_integration/preparation_events.md` for its scope.

The artifact was relocated byte-for-byte after the serving context checker
interpreted its fixture context as an advertised serving cap. Original SHA256:
`c5aec0171f39c95b35bbd59b98a3391ea557a7fe4b190949f3efbe361c3b7a3e`.
The original is also retained in commit `425000e4bd982a48db43a6963e03359d07b509b0`
at `readiness_vllm/preparation_events_host_check.json` under this model directory.
No numeric values or event records were changed.
