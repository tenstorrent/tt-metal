# Source provenance

This implementation originates from **Samuel Jett** ([sjettTT](https://github.com/sjettTT),
sjett@tenstorrent.com), in
[sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next)
at immutable commit `cd9a11771107ea2c27da3303a0556ff7343e4af5`.
That snapshot contains 452 tracked files in this model subtree. The original
source checkout and its measurement records are preserved separately from the
publication port and its newly measured results.

The port targets tt-metal `bdfc59036eea3e988ba0e2374c12ca0c15c6c970` and depends
on Samuel Jett's original [tt-metal PR #57564](https://github.com/tenstorrent/tt-metal/pull/57564),
head `df54becdb774e4b9ac21d664e7eb346ebe91d4d6`. Compact expert packing belongs
to that original prerequisite. The publication stack carries its six commits
with their original author attribution; it does not present that work as an
independent new contribution.

Port adaptations preserve the model's numerical policy while using current
trace-allocation APIs and the current native build. The shared MoE streaming
pipeline is an explicit model opt-in, so existing operation callers keep their
serial pipeline and buffer sizes. The vLLM adapter declares its fabric settings
through model capabilities before device creation, using the existing
`EXTRA_MODELS_DIR` route in plugin commit
`1d87a00e7d91ec246582d07865ec4f8b0a8fb25c`.

The serving adapter has one resident slot and no MTP. Its `decode_only` sampling
route draws on the CPU from reduced model outputs; requests needing full logits
use the plugin's host sampler. Standalone MTP measurements describe a separate
execution mode and are not vLLM throughput.

Historical measurements and proposal pins shipped with the source are source
evidence, not validation passes for this port. In particular, the plain slab
baseline predates the source's default `gr_recip_last` policy, while the newer
chunked and MTP baselines include it. Configuration, policy, sample count, EOS
handling, and reference coverage must accompany comparisons.
