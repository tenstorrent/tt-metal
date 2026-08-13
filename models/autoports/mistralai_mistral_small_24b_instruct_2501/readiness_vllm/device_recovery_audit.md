# Device recovery audit

Date: 2026-08-13 UTC

The final-source rerun began with no `run_vllm_server`, `vllm.entrypoints.openai.api_server`, or `VLLM::EngineCore` owner. A bounded `tt-smi -ls --local` completed and listed UMD chips 0-3 (four Blackhole p300c devices). During earlier debugging, one reset was mistakenly started before the prior runner had fully released device ownership; that runner was then stopped explicitly, and all subsequent resets used a process-owner check first.

The first bounded 1x4 mesh smoke failed while opening the mesh:

```text
Device 0: Timed out while waiting for active ethernet core 29-25 to become active again.
Heartbeat changed: false
```

With no device-owning serving process, recovery used one bounded reset:

```bash
timeout 180 /home/mvasiljevic/.ttsmi-venv/bin/tt-smi -r
timeout 60 /home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls --local
```

Reset returned within the bound. The post-reset list again showed chips 0-3. The required mesh-open proof then passed:

```bash
TT_LOGGER_LEVEL=fatal timeout 120 python_env/bin/python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK devices=4 shape=1x4')
PY
```

Result: `MESH_SMOKE_OK devices=4 shape=1x4`, exit status 0. No second reset or lock clearing was needed. The final production rerun resumed only after this proof.

After all final production evidence and the 72-pass/1-skip compatibility profile, the owning runner stopped the API server and EngineCore cleanly. A fresh `pgrep` found no serving owner and `tt-smi -ls --local` again listed chips 0-3. The first post-teardown mesh smoke reproduced the device-0 ERISC heartbeat timeout. Because process ownership was clear, one bounded `timeout 180 .../tt-smi -r` was run. The following list showed all four boards, and `open_mesh_device(MeshShape(1, 4), trace_region_size=0)` opened and closed successfully. No serving process or open mesh was left behind.
