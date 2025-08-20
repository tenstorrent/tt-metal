## Program cache review — data_movement/split

Status: Issue found — cache-hit override sets second output address incorrectly.

Findings
- Old/type-erased tiled split into two outputs; program factory provides override callback.
- Hashing: default determinants include split axis/strategy via compile-time args and shapes; runtime addresses excluded.
- Override bug: `dst_1_dram_buffer` is derived from `output_tensors.at(0)` instead of `output_tensors.at(1)` in the override lambda, so both writer args point to the first output on cache-hit.
  - Location: `ttnn/cpp/ttnn/operations/data_movement/split/device/split_program_factory.cpp:L218-L220`.
- A failing two-run cache test is added under `program_cache/data_movement/split/failures/...` that exposes a PCC mismatch on the second run.
