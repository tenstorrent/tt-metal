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

#### Model tests
CI tests related to models pass (choose "models-mandatory", "models-extended" presets, or select applicable tests manually):
- [ ] [(Single) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select.yaml) 
  - [ ] models-mandatory
  - [ ] models-extended
  - [ ] other selection - specify
- [ ] [(T3K) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-t3k.yaml)
  - [ ] models-mandatory
  - [ ] models-extended
  - [ ] other selection - specify
- [ ] [(Galaxy) Choose your pipeline](https://github.com/tenstorrent/tt-metal/actions/workflows/pipeline-select-galaxy.yaml)
  - [ ] models-mandatory
  - [ ] models-extended
  - [ ] other selection - specify
