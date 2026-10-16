# Source provenance

This implementation is ported from the source maintained and optimized by
**Samuel Jett** ([sjettTT](https://github.com/sjettTT), sjett@tenstorrent.com), in
[sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next)
at immutable commit `cd9a11771107ea2c27da3303a0556ff7343e4af5`.
That snapshot contains 452 tracked files in this model subtree. The original
source checkout and its measurement records are preserved separately from the
publication port and its newly measured results.

The initial source export, commit `2f6fcf1a6dccf91cf480c66dab31afe25f7729c3`,
records both Author and Committer as `Codex <codex@openai.com>`. This export
attribution is distinct from Samuel Jett's source maintenance, subsequent
optimizations and original operation commits retained below.

The current integration starts from tt-metal `3407283c1ca1998d95cf576c8f6130a8b8f9d5cc`.
Samuel Jett's compact packing [#57564](https://github.com/tenstorrent/tt-metal/pull/57564)
and extent-one local output [#57448](https://github.com/tenstorrent/tt-metal/pull/57448)
are merged upstream. His replacement packed routing [#60469](https://github.com/tenstorrent/tt-metal/pull/60469)
at `07dc40e532e51fc74a8caba37852d3d208543494` and expert rows
[#59945](https://github.com/tenstorrent/tt-metal/pull/59945) at
`70f967df3c8d221be5ef79d979cd3d9f986d0bed` retain their original author history.
The retired pipeline [#58476](https://github.com/tenstorrent/tt-metal/pull/58476) is closed;
its kernels and launch switches are not part of this integration.

The model selects the public local-output path with `cluster_axis=0` on its 1x4 mesh.
That path writes owned expert rows into the caller's persistent output. The model's
existing fill or owner-masked reduction controls unowned rows. Expert-row placement
currently requires the fused combine, so the integration keeps SingleCluster calls
on the ring writer; it does not reinterpret them as single-device fused calls.
This compatibility choice is not a measured performance result. Build and native
qualification of the integrated revision remain required.

Applicable public GDN callers reuse Izajasz Wrosz's merged upstream
[PR #57440](https://github.com/tenstorrent/tt-metal/pull/57440), commit
`96cc4a7937f19ee205717577fcf4d10269d043b4`. The port does not add another public
prep/scan binding. Raw flat query/key and multi-lane inputs retain that public
route and their existing scale contract.

The frozen model's normalized prefill producers instead use the model-owned
`gdn_source_chunk` policy through the public `ttnn.generic_op` API. Its six
kernels retain Samuel Jett's exact source bytes and original preparation,
packing, readout and state-update sequence. The shared GDN operation passes
independent numerical controls, but its combined arithmetic changes words
required by the frozen source gate. The model policy keeps the admitted query
scale and normalization without changing shared operation defaults or kernels.
Its source-word and independent reference controls cover complete 32-row
chunks at 32, 128, 2048 and 4096 rows, zero/nonzero states and masked commits.
These operator results do not qualify whole-model source trajectories, serving,
task quality or another hardware topology.

Port adaptations preserve the model's numerical policy while using current
trace-allocation APIs and the current native build. The native operation selects its supported implementation; the model does not
request private replay-ring or pipeline variants. The vLLM adapter declares its fabric settings
through model capabilities before device creation, using the existing
`EXTRA_MODELS_DIR` route in plugin commit
`1d87a00e7d91ec246582d07865ec4f8b0a8fb25c`.

The serving adapter has one resident slot and no MTP. Its `decode_only` sampling
route samples the full-vocabulary row on the CPU and returns only the selected
token ID to the plugin; requests needing full logits return that row to the
plugin's host sampler. Standalone MTP measurements describe a separate
execution mode and are not vLLM throughput.

Historical measurements and proposal pins shipped with the source are source
evidence, not validation passes for this port. In particular, the plain slab
baseline predates the source's default `gr_recip_last` policy, while the newer
chunked and MTP baselines include it. Configuration, policy, sample count, EOS
handling, and reference coverage must accompany comparisons.
