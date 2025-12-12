### Ticket
Link to Github Issue

### Problem description
Provide context for the problem.

### What's changed
Describe the approach used to solve the problem.
Summarize the changes made and its impact.

### Checklist

- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI with demo tests passes (if applicable)
- [ ] [cpp-unit-tests](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml) job passes
- [ ] New/Existing tests provide coverage for changes


```
models-mandatory: depending on the platform:
- T3K: unit tests
- Galaxy: quick tests

models-extended: runs all respective tests from the above models-mandatory preset AND:
- T3K: demo, perf-models tests
- Galaxy: demo, perf-models tests
```

#### Model tests
CI tests related to models pass (choose "models-mandatory", "models-extended" presets, or select applicable tests manually):
- [ ] [(Single) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml) 
  - [ ] `models-mandatory` preset (runs: 
      [Device perf regressions](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-device-models.yaml), 
      [Frequent model and ttnn tests](https://github.com/tenstorrent/tt-metal/actions/workflows/fast-dispatch-full-regressions-and-models.yaml) and
      [(internal) C++ unit tests](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/cpp-post-commit.yaml))
  - [ ] `models-extended` preset (runs: 
      the mandatory tests, plus [Demo](https://github.com/tenstorrent/tt-metal/actions/workflows/single-card-demo-tests.yaml)
      and [Model perf tests](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-models.yaml)
    )
  - [ ] other selection - specify runs
- [ ] [(T3K) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-t3k.yaml)
  - [ ] `models-mandatory` preset (runs: 
      [Unit tests](https://github.com/tenstorrent/tt-metal/actions/workflows/t3000-unit-tests.yaml))
  - [ ] `models-extended` preset
      (runs: 
      the mandatory tests, plus [Demo](https://github.com/tenstorrent/tt-metal/actions/workflows/t3000-demo-tests.yaml) and 
      [Model perf tests](https://github.com/tenstorrent/tt-metal/actions/workflows/t3000-model-perf-tests.yaml))
  - [ ] other selection - specify runs
- [ ] [(Galaxy) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-galaxy.yaml)
  - [ ] `models-mandatory` preset (runs: 
      [Quick tests](https://github.com/tenstorrent/tt-metal/actions/workflows/galaxy-quick.yaml))
  - [ ] `models-extended` preset 
      (runs: 
      the mandatory tests, plus [Demo](https://github.com/tenstorrent/tt-metal/actions/workflows/galaxy-demo-tests.yaml) and 
      [Model perf tests](https://github.com/tenstorrent/tt-metal/actions/workflows/galaxy-model-perf-tests.yaml))
  - [ ] other selection - specify runs
