# Buffered token-out diagnosis

Failing contract: fixed-step token-out generation should replay nonblocking traces with device token/position feedback and no per-token host readback.

Source observation: `tt/generator.py::generate` invokes `_read_tokens()` immediately after every `_replay()`. `_read_tokens` calls `ttnn.to_torch`, forcing output readiness before Python can submit the next replay. `_replay` already writes common-sampler output directly into `self.tokens`, and `_forward` advances both device position inputs. The readback is unnecessary when `stop_on_eos=False` and `next_input=None`.

Hypothesis: recording each sampled token into device storage and reading the completed sequence once preserves exact fixed-step output while removing the per-token host dependency. EOS stopping and teacher forcing still require their existing caller interaction.

Chosen primitive: `ttnn.indexed_fill` on row-major uint32 `[steps,1,1,32]`, indexed along dimension 0 by one device uint32. Its source validation accepts rank-4 row-major input and a runtime integer index; the dataflow kernel reads the index on device. Since it returns a new output, capture `indexed_fill -> copy back to persistent storage -> plus_one(index)` in a dedicated warmed trace. Scratch allocation occurs during warm/capture, never replay. The buffer remains stable across same-shaped requests; request boundaries reset the index.

Discriminating checks: exact buffered-vs-unbuffered greedy token equality, repeated request reuse, short/tail generation lengths, device index final value, final token feedback identity, and exactly one end-of-window token readback with zero steady-state token/position/page-table refreshes. A standalone recorder probe with changing synthetic uint32 values checks indexing before any expensive model run. No hardware result is claimed by this source diagnosis.
