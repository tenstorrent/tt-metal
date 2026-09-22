# Longer-context prefill scope

The initial shared-runner acceptance is 2K capacity with two independent slots.
After that passes, the follow-up list covers 4K, 8K, 16K, 32K and 64K. Keep SP4/TP8,
32 layers, BFP8 K/V and 1,024-token compute chunks at every capacity.

For each length, fill both slots, verify source-table boundary reads and slot
isolation, check completion/shutdown, and record runtime. Full golden K/V comparison
at every larger length is optional and must be requested separately.

[Existing model performance results](docs/performance-prefill.md) cover the full
embedding/layer-stack/final-normalization/vocabulary-head path through 64K. Those
saved measurements are distinct from acceptance of the shared runner and producer.

[SC1 reproduction commands](docs/runner-integration.md) define the current 2K gate.
128K is deferred.
