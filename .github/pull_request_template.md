### Ticket
Link to Github Issue

### Problem description
Provide context for the problem.

### What's changed
Describe the approach used to solve the problem.
Summarize the changes made and its impact.

### Checklist

- [ ] [![All post-commit tests](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml?query=branch:{{branch_name}})
- [ ] [![Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml?query=branch:{{branch_name}})
- [ ] [![cpp-unit-tests](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml?query=branch:{{branch_name}})
- [ ] New/Existing tests provide coverage for changes


#### Model tests

If your changes cover model-related code, you should run tests corresponding to affected models and platforms (Single card, T3K, Galaxy). "Choose your pipeline" workflows facilitate running multiple kinds of tests in a single run. Each offers `models-mandatory` and `models-extended` presets.
The former includes a minimal set of tests, to be run always. The latter extends that with additional ones - use your best judgement in deciding which is the most appropriate for your PR.

- [ ] [![(Single) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml?query=branch:{{branch_name}})
  - [ ] `models-mandatory` preset (runs: [Device perf regressions](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-device-models.yaml) and [Frequent model and ttnn tests](https://github.com/tenstorrent/tt-metal/actions/workflows/fast-dispatch-full-regressions-and-models.yaml))
  - [ ] `models-extended` preset (runs: the mandatory tests, plus [Demo](https://github.com/tenstorrent/tt-metal/actions/workflows/single-card-demo-tests.yaml) and [Model perf](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-models.yaml) tests)
  - [ ] other selection - specify runs

- [ ] [![(T3K) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-t3k.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-t3k.yaml?query=branch:{{branch_name}})
  - [ ] `models-mandatory` preset (runs: [Unit tests](https://github.com/tenstorrent/tt-metal/actions/workflows/t3000-unit-tests.yaml))
  - [ ] `models-extended` preset (runs: the mandatory tests, plus [Demo](https://github.com/tenstorrent/tt-metal/actions/workflows/t3000-demo-tests.yaml) and [Model perf](https://github.com/tenstorrent/tt-metal/actions/workflows/t3000-model-perf-tests.yaml) tests)
  - [ ] other selection - specify runs

- [ ] [![(Galaxy) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-galaxy.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-galaxy.yaml?query=branch:{{branch_name}})
  - [ ] `models-mandatory` preset (runs: [Quick tests](https://github.com/tenstorrent/tt-metal/actions/workflows/galaxy-quick.yaml))
  - [ ] `models-extended` preset (runs: the mandatory tests, plus [Demo](https://github.com/tenstorrent/tt-metal/actions/workflows/galaxy-demo-tests.yaml) and [Model perf](https://github.com/tenstorrent/tt-metal/actions/workflows/galaxy-model-perf-tests.yaml) tests)
  - [ ] other selection - specify runs
