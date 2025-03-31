### Ticket
Link to Github Issue

### Problem description
Provide context for the problem.

### What's changed
Describe the approach used to solve the problem.
Summarize the changes made and its impact.

### Checklist
- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Model regression](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-models.yaml) CI passes (if applicable)
- [ ] [Device performance regression](https://github.com/tenstorrent/tt-metal/actions/workflows/perf-device-models.yaml) CI passes (if applicable)
- [ ] **(For models and ops writers)** Full [new models tests](https://github.com/tenstorrent/tt-metal/actions/workflows/full-new-models-suite.yaml) CI passes (if applicable)
- [ ] New/Existing tests provide coverage for changes
