# Prefix-cache source diagnosis, 2026-10-19

Scope: source-only investigation of experiment head `c56f74cc7f` and downloaded run `39230676369` artifacts. No local tests, imports, builds, servers, device operations, or validation scripts were run. Model source was not edited. The isolated investigation bypassed the AutoDebug launcher under the user's no-local-execution constraint.

## Finding

The strongest supported explanation is a prefill/decode numerical-contract mismatch, not a demonstrated position error. This remains a hypothesis until the remote experiment below separates arithmetic from cache ownership. A cache hit alone does not establish that the reused physical pages contain the correct layer's KV.

`live.xml` records a 33-token prompt `[100..132]`, a hit delta of 32, cold token 128 and warm token 130. It does not retain their complete logits or margins, so neither the magnitude nor the first divergent layer is known. This synthetic sequence is not evidence that the discrepancy is harmless. The cold/warm timings include different cold compilation costs and do not prove acceleration.

## Source evidence

- `tt/generator.py:441` validates absolute end lengths and slices `tokens[start:length]`. For start 32 and end 33 this uploads only token 132. `tt/model.py:218` forwards start 32 unchanged to every layer and selects the last suffix row for logits.
- `tt/multichip_decoder.py:1179` uses `position = start_pos + offset` for both uint32 RoPE lookup and int32 cache position. It selects the request's page-table row, retaining logical page indices. Thus the only suffix token uses absolute position 32, logical page 1, row 0. There is no visible off-by-one or accidental suffix-relative position here.
- `tt/functional_decoder.py:111` pads the cold 33-token sequence to 64 rows. `tt/optimized_decoder.py:1376` fills exactly 64 KV rows, casting to the cache dtype. It then computes first-chunk attention directly from BF16 K/V, **not the quantized BFP8 cache**. Causal attention masks rows 33..63. Prefix rows 0..31 form a complete independent page; the warm update at row 32 should not touch that page. Partial-page preservation is therefore not the leading explanation for this first failure.
- Warm continuation invokes each layer's decode path. It reads the BFP8 prefix cache and uses decode arithmetic. `config/precision.json` explicitly differs between modes: QKV LoFi decode versus HiFi4 prefill, shared weights BFP8 decode versus BF16 prefill, and sliding expert-down weights BFP8 decode versus BFP4 prefill. SDPA fidelity and EP/TP implementation also differ. These are material algorithmic differences, not merely a different batch size. Even perfect page reuse cannot guarantee identical cold-prefill logits with this implementation.
- The transient sliding-prefill tail is not used by continuation; continuation reads paged cache. At position 32 there is no sliding-window eviction. Tail reset/window rollover is accordingly a lower-priority hypothesis.
- `tt/generator_vllm.py:114` pools by `(tensor_index, cache_dtype)` and aliases equal-byte tile storage. Per-device sliding pages `[2,32,256]` and full pages `[1,32,512]` each have 16 tiles; BFP8 tiles have equal storage size, so the shapes have equal physical page bytes. The scheduler spec deliberately uses BF16 accounting with equal padded page sizes. This is not itself evidence of corruption, but correctness depends on scheduler block IDs being disjoint where layers share a tensor pool and stable for every reused group. The artifact does not prove that invariant.

## Discriminating remote experiment

Use the selected full 30-layer policy and exactly the 33-token failing prompt. Retain complete BF16 logits, top candidates/margins, and per-layer suffix outputs. Use independent cache copies or restore cache bytes between arms.

1. Cold-prefill all 33 tokens into dedicated caches and retain logits plus each layer's KV rows 0..31.
2. On those same physical page mappings, replay only token 132 at position 32 through ordinary model decode; separately replay through `prefill_forward(start_pos=32)`. Neither arm uses scheduler cache reuse. They should agree with each other. If both reproduce token 130, the arithmetic-path mismatch is sufficient to explain the API failure.
3. Repeat the continuation with the scheduler's warm HMA tables. Log each layer's pool identity, physical block IDs for logical pages 0 and 1, and prefix KV checksums before processing the suffix. Compare logical prefix rows against arm 1. A mismatch before suffix computation establishes an ownership/lifetime problem independently of numerical differences.
4. If the prefix bytes match but the logits differ between arms 2 and 3, compare the selected table row, absolute positions, and first divergent layer; also check padded SDPA page entries are valid and not aliases of writable pages. If arms 2 and 3 agree, investigate precision/operation parity instead of altering positions or HMA mappings.

An additional BF16-cache arm can estimate cache quantization's contribution but cannot alone eliminate mode-specific weight, fidelity, expert, or RoPE differences. Do not loosen the API equality assertion or remove the seed on the strength of this source diagnosis. Either establish an explicit acceptable numerical contract, or implement continuation with arithmetic matching prefill and validate remotely.
