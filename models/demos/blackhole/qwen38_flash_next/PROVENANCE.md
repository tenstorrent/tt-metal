# Provenance

## Source

Ported from [sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next) at commit
`cd9a11771107ea2c27da3303a0556ff7343e4af5`, written and maintained by **Samuel Jett**
([sjettTT](https://github.com/sjettTT), sjett@tenstorrent.com). The source's initial export commit,
`2f6fcf1a6dccf91cf480c66dab31afe25f7729c3`, records both Author and Committer as `Codex <codex@openai.com>`; the
maintenance, optimizations and operation commits after it are Samuel Jett's.

Not ported: the source's fused kernels `qsa_score_pages` (32-row verify tile), `moe_combine`, `gdn_prefill_rows` and
`sparse_sdpa_tiled`. The op chains they replaced run instead, and their `QWEN38_FUSED` / `QWEN38_FUSED_OFF` names
are not registered.

## Upstream work used

- Samuel Jett's tt-metal changes [#57564](https://github.com/tenstorrent/tt-metal/pull/57564) (compact expert
  packing) and [#57448](https://github.com/tenstorrent/tt-metal/pull/57448) (extent-one local output for
  `moe_compute`), both merged.
- Izajasz Wrosz's public GDN operation, [#57440](https://github.com/tenstorrent/tt-metal/pull/57440) (commit
  `96cc4a7937f19ee205717577fcf4d10269d043b4`): `ttnn.transformer.chunk_gated_delta_rule` runs every GDN chunk
  recurrence (the 32- and 128-row prefill chunks, the slab, the verify-rows wrap and the lanes with the fold off). Its
  arithmetic differs from the source's in the last bits, so greedy streams are not the source's word for word.

## Derived code

- `ttnn/fused/kernels/source_reciprocal.h`: the source's
  `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h`, verbatim apart from names and
  formatting.
- `ttnn/fused/kernels/source_softplus.h`: the source's softplus arithmetic, adapted from tt-metal's
  `ckernel_sfpu_gelu.h` tail.
- `tests/data/source_sfpu_rounding/`: fixtures recorded on the source commit.

## Licensing

Apache-2.0, as the rest of tt-metal. The SPDX headers name Tenstorrent AI ULC or Tenstorrent USA, Inc.;
`ttnn/fused/kernels/source_reciprocal.h` also names `© 2025 Jason Davies`. No model weights are included:
`tools/download_checkpoint.py` fetches the pinned checkpoint revision.
