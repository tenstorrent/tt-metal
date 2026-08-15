# AutoDebug: streaming prefill uses a fixed chunk boundary incompatible with vLLM's bound KV page

## Finding

The failure is a deterministic contract mismatch, not an invalid prompt. The model streams at a fixed 32,768-token stack chunk, while the vLLM adapter can rebind the full-attention KV cache to an externally supplied page size. In the failing run that effective page size is 800, so the second chunk starts at 32,768 and fails because `32768 % 800 == 768`.

Exact evidence:

- `readiness_vllm/server.log:15` starts vLLM with requested scheduler `block_size=64`, but `server.log:20` explicitly says, `Setting attention block size to 800 tokens to ensure that attention page size is >= mamba page size`; line 21 says the mamba and attention page sizes are made exactly equal. This directly establishes the effective hybrid attention page as 800.
- `readiness_vllm/server.log:289` records `prompt_token_ids_len=32780`, 41 `block_ids` (`127..167`), and 32,780 scheduled tokens. This independently agrees with the logged 800-token page (`ceil(32780/800) == 41`), and the in-tree adapter test explicitly models `page_size = 800` (`tests/test_vllm_adapter_contract.py:20`).
- `tt/generator_vllm.py:141-155` takes `serving_page_size = kv_cache_shape[2]`; when it differs from 64 it changes `gen.model.page_size` and rebuilds the page table. `tt/generator_vllm.py:160-164` also constructs the replacement full-attention caches from that external shape.
- `tt/model.py:80` fixes `PREFILL_STACK_CHUNK_SIZE = 32768`; `tt/model.py:488-499` iterates at that stride and writes `layer._prefill_chunk_start = start`.
- `tt/multichip_decoder.py:1039-1044` independently derives `page_size` from the actually bound cache (`self.caches["key"].shape[2]`) and rejects any non-page-aligned nonzero `chunk_start`. Thus the first chunk at zero succeeds and the second at 32,768 raises exactly the logged exception at `server.log:338-354`.
- Existing tests cover the fixed constant and 32-token linear metadata (`tests/test_vllm_adapter_contract.py:20,42-52`) but do not exercise stack streaming with a rebound page size.

## Ranked hypotheses

1. **Confirmed by the causal source chain: fixed stack stride versus rebound 800-token KV page.** It predicts the precise first failing boundary and exception. `32768 % 64 == 0`, explaining why model-owned/default caches do not expose it; `32768 % 800 != 0`, explaining vLLM hybrid-cache failure.
2. **Possible secondary bug: the external cache page axis is being inferred incorrectly.** `allocate_kv_cache()` assumes `kv_cache_shape[2]` is the page/token axis. Current cache construction and the observed 41-page allocation are consistent with 800, so this is lower-ranked. A host contract test should assert the expected vLLM shape layout rather than silently assume it.
3. **Refuted as primary cause: logical prompt nonalignment.** The exception tests `chunk_start`, not prompt length. A 32,780-token prompt merely crosses the second fixed stack chunk. Aligning benchmark input/context would hide a valid-input bug and is not a fix.

## Smallest verify/refute experiment (host-only)

Add a pure unit test for a helper that selects stack chunk size from `(max_chunk=32768, page_size, linear_chunk=32)`, without importing or executing TT hardware:

```python
assert choose_chunk_size(32768, 64, 32) == 32768
assert choose_chunk_size(32768, 800, 32) == 32000
for page_size in (64, 800, 1024):
    chunk = choose_chunk_size(32768, page_size, 32)
    assert 0 < chunk <= 32768
    assert chunk % page_size == 0
    assert chunk % 32 == 0
```

Then statically simulate `range(0, sequence, chunk)` for sequence 32,780 and assert every nonzero start is page-aligned. This refutes hypothesis 1 if the live adapter's captured `kv_cache_shape[2]` is not 800 or if layer caches do not retain that shape; add one initialization-time diagnostic/assertion comparing `gen.model.page_size` with every full-attention `layer.caches["key"].shape[2]` to settle that without a long prompt.

## Minimal intervention

At the streaming owner (`Qwen36Model._prefill_forward_streaming`), derive a request/model stack stride that is no larger than 32,768 and is a multiple of both the effective KV page size and `LINEAR_PREFILL_CHUNK_SIZE`. Since accepted serving pages are tile multiples, for 800 this is simply `floor(32768 / 800) * 800 == 32000`. Iterate and slice using that stride instead of the class constant. Keep the decoder guard: it is valuable validation. Do not pad the logical prompt, alter scheduler block tables, or require benchmark alignment.

The helper must reject a page/alignment quantum larger than the maximum chunk rather than returning zero. Also verify that all full-attention layers share the same bound page size; otherwise selecting from `model.page_size` alone could mask inconsistent cache binding.

## Boundary regression matrix

Run the host range/alignment test for effective pages 64 (default), 800 (observed hybrid), and 1024. For page 800 the selected stride is 32,000; cover logical prompt lengths `31999, 32000, 32001, 32767, 32768, 32769, 32780, 33599, 33600, 33601, 63999, 64000, 64001`. Assert: all are accepted; terminal-logit selection uses the logical length; every nonzero chunk start is divisible by 800; metadata slices remain 32-token aligned; page-table slicing includes `ceil((start + sequence)/page_size)`; and request state is cleared after completion. Repeat the immediate `stride-1/stride/stride+1` cases for pages 64 and 1024. Include mixed slots with one zero-length inactive slot because that path substitutes `cache_page_table`.

## Exact server rerun after the fix

Start the same server configuration recorded by the failing log (Qwen/Qwen3.6-27B, `--max-model-len 262144 --block-size 64 --max-num-seqs 1`, TT `sample_on_device_mode=all`, trace region 200000000, and `FABRIC_1D_RING`). Then run:

```bash
/home/mvasiljevic/tt-metal/python_env/bin/vllm bench serve \
  --backend openai-chat --model Qwen/Qwen3.6-27B \
  --base-url http://localhost:8000 --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 32768 --random-output-len 128 \
  --num-prompts 1 --max-concurrency 1 --ignore-eos --temperature 0.0
```

Pass criteria: chat rendering still reports 32,780 prompt tokens, the request completes 128 output tokens, no page-boundary exception occurs, and the server remains healthy for a subsequent short request. Also rerun input lengths 31,988 and 32,789 if exact rendered-token control is needed to bracket the 32,000 effective boundary (the observed template overhead is 12, but validate the rendered length in the scheduler dump).

No hardware or live-server commands were run during this investigation, and no implementation or test files were changed.
