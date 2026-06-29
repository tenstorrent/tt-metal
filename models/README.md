# TT-Metalium / TT-NN Models

## Model Demos

- [Demo models on Wormhole](../README.md#wormhole-wh-models)
- [Demo models on TT-QuietBox & TT-LoudBox (Wormhole)](../README.md#tt-quietbox--tt-loudbox-2x4-mesh-of-whs-models)
- [Demo models on Single Galaxy (Wormhole)](../README.md#single-galaxy-8x4-mesh-of-whs-models)

## Latest Releases

Visit the [releases](https://github.com/tenstorrent/tt-metal/tree/main/releases) folder for details on releases, release notes, and estimated release dates.

## Release methodology

- Models that are advertised as part of release, usually the demo models, are treated as first-class citizens, and therefore are treated as tests.
- Model writers are responsible for ensuring their demo model tests are always passing. Any failure is treated highest-priority (or P0) failure.
- Model writers are responsible for advertising which release tag (including release candidates) contains passing tests for their demo models.
- Model writers are responsible for updating their perf metrics for the demo models at a regular cadence. Currently, the cadence is at least every 2 weeks.
