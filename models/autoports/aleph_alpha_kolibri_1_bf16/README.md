# Kolibri-1-BF16 TTNN and vLLM serving

> **About this branch.** This is the lean CI/serving import of the agentic
> bring-up (tenstorrent/tt-agentic-bringup-qb2#73): `tt/`, `tests/`, the
> selected precision config and the context contract. The stage evidence the
> text below links under `doc/` (68 GB) is **not** on this branch; it lives in
> the bring-up checkout and is summarized in the tracking issue. Two
> deliberate deviations from the pipeline output: the served context is the
> native **262,144** tokens, not the 1,048,576 quoted below (owner decision,
> tt-agentic-bringup-qb2#78), and the checkpoint is read from a TTI-mounted
> `MODEL_WEIGHTS_DIR`/`HF_MODEL` directory when present. Serving needs the
> matching vllm-tt-plugin branch (model and `kolibri1` config registration,
> `kolibri1` reasoning and tool parsers).

**Primary vLLM P128/G128/N1, concurrency 1, server 1 slot: 204.635 ms TTFT and
62.8327 decode tokens/s/user** on QB2/P300x2, 1×4 Blackhole, with the full
1,048,576-token served context. The matched baseline is 219.746 ms and
62.8410 tokens/s/user: TTFT improves 6.88%, with decode essentially unchanged.
See [optimized serving evidence](doc/optimized_vllm/README.md) for raw metrics,
secondary CI capacity results, and gate/review evidence. All serving gates pass;
independent [final review](doc/optimized_vllm/stage_review_final.md) returned **clean-pass**.


The current precision default is **BFP4/LoFi decoder weights, BFP8/HiFi2 head,
BF16 KV cache and logits**. It passes the 100-token AIME24 chat-template gate
at **95% prefill / 96% teacher-forcing top-1**, with **100% top-5 and top-100**.

Post-selection warmed token-out performance is **62.9843 tokens/s/user** with
**266.775 ms TTFT** on QB2/P300x2 (four Blackhole chips, 1×4 mesh): batch 1,
128 input /128 generated tokens, full 1,048,576-token cache allocation, median
of five no-readback decode loops. Later serving comparisons must use
[the post-selection performance artifact](doc/datatype_sweep/perf_summary.json).
The separate traced teacher-forcing selection metric is 62.1534 tokens/s/user
(P197/G100); a fresh default construction reproduces 62.1477 tokens/s/user.

Pinned model: `Aleph-Alpha/Kolibri-1-BF16`, revision
`7a8f290e7858825c3cf5e4c447ba68345de9f1d3`.

The complete 50-layer model and generator are in [tt/model.py](tt/model.py)
and [tt/generator.py](tt/generator.py). Both consume the required
[selected precision artifact](doc/datatype_sweep/selected_precision_config.json)
by default. An explicit `precision_config` or `KOLIBRI_PRECISION_CONFIG`
overrides it; [the baseline config](doc/datatype_sweep/configs/baseline_bfp4_lofi.json)
reproduces the historical Stage 7 precision policy.

Stage 8 full-context, non-aligned and B32 checks pass, along with allocation-tracked B1/B2/B31/B32 replay contracts. Independent review returned [clean-pass](doc/datatype_sweep/stage_review_final.md). Stage 8 is complete; local checkpoint SHAs are in the work log.
The [vLLM adapter](tt/generator_vllm.py) uses the normal generator construction
contract, the standalone TT plugin, persistent async traces, and on-device
split sampling. [Stage 9 integration](doc/vllm_integration/README.md) is complete.

See the [datatype-sweep report and charts](doc/datatype_sweep/README.md),
[work log](doc/datatype_sweep/work_log.md),
[context contract](doc/context_contract.json), and
[runtime ownership contract](doc/full_model/runtime_contract.md).
Historical implementation and optimization evidence remains in
[Stage 6](doc/full_model/README.md) and [Stage 7](doc/optimized_full_model/README.md).
