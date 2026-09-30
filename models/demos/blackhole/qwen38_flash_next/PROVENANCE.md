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

The port targets tt-metal `bdfc59036eea3e988ba0e2374c12ca0c15c6c970` and depends
on Samuel Jett's original [tt-metal PR #57564](https://github.com/tenstorrent/tt-metal/pull/57564),
head `df54becdb774e4b9ac21d664e7eb346ebe91d4d6`. Compact expert packing belongs
to that original prerequisite. The publication stack carries its six commits
with their original author attribution; it does not present that work as an
independent new contribution.

The MoE ancestry also retains Samuel Jett's original
[PR #57448](https://github.com/tenstorrent/tt-metal/pull/57448), head
`ae0691f1cf3fc4c227bc0a0f48a43959589b09fc`: four source-ring and LocalOutput
commits, separate from the six compact-packing commits. Both original PRs
remain prerequisites. The additional packed-token, streaming and replay work
is scoped after them; their existing contributions are not duplicated in a
new prerequisite review. The source's explicit local-combine and idle-expert
handling remains part of the additional port.

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
trace-allocation APIs and the current native build. The shared MoE streaming
pipeline is an explicit model opt-in, so existing operation callers keep their
serial pipeline and buffer sizes. The vLLM adapter declares its fabric settings
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
