<!-- toc -->

Table of Contents
=================

   * [Contributing to tt-metal](#contributing-to-tt-metal)
   * [Machine setup](#machine-setup)
      * [Hugepages setup](#hugepages-setup)
   * [Developing tt-metal](#developing-tt-metal)
      * [Setting up Git](#setting-up-git)
      * [Setting logger level](#setting-logger-level)
      * [Building and viewing the documentation locally](#building-and-viewing-the-documentation-locally)
      * [Cleaning the dev environment with make nuke](#cleaning-the-dev-environment-with-make-nuke)
   * [Running tests on tt-metal](#running-tests-on-tt-metal)
      * [Running pre/post-commit regressions](#running-prepost-commit-regressions)
      * [Running model performance tests](#running-model-performance-tests)
   * [Debugging tips](#debugging-tips)
   * [Contribution standards](#contribution-standards)
      * [File structure and formats](#file-structure-and-formats)
      * [CI/CD Guidelines](#cicd-guidelines)
      * [Documentation](#documentation)
      * [Git rules and guidelines](#git-rules-and-guidelines)
      * [Code reviews](#code-reviews)
      * [New feature and design specifications](#new-feature-and-design-specifications)
      * [Release flows](#release-flows)
      * [Logging, assertions, and exceptions](#logging-assertions-and-exceptions)
   * [Hardware troubleshooting](#hardware-troubleshooting)
      * [Resetting an accelerator board](#resetting-an-accelerator-board)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

<!-- tocstop -->

## Contributing to tt-metal

Thank you for your interest in this project.

If you are interested in making a contribution, then please familiarize
yourself with our technical contribution standards as set forth in this guide.

Next, please request appropriate write permissions by [opening an
issue](https://github.com/tenstorrent-metal/tt-metal/issues/new/choose) for
GitHub permissions.

All contributions require:
- an issue
  - Your issue should be filed under an appropriate project. Please file a
    feature support request or bug report under Issues to get help with finding
    an appropriate project to get a maintainer's attention.
- a pull request (PR).
  - Your PR must be approved by appropriate reviewers.

Furthermore, all PRs must follow the [contribution
standards](#contribution-standards).

## Machine setup

### Hugepages setup

Hugepages is required to both run and develop on the Metal project.

If you ever need to re-enable Hugepages, you can try the script we homemade
for this:

```
sudo python3 infra/machine_setup/scripts/setup_hugepages.py enable
```

Then to check if Hugepages is enabled:

```
python3 infra/machine_setup/scripts/setup_hugepages.py check
```

## Developing tt-metal

Currently, the most convenient way to develop is to do so on our cloud
machines. They have prerequisite dependencies, model files, and other settings
set up for users.

Please refer to the [README](README.md) for source installation and environment
setup instructions, then please read the the [developer's
page](docs/source/dev_onboarding/get_started.rst).

### Setting up Git

We use `#` as a special character to denote issue numbers in our commit
messages. Please change your comment character in your Git to not conflict with
this:

```
git config core.commentchar ">"
```

### Setting logger level

In order to get debug level log messages, set the environment variable
`TT_METAL_LOGGER_LEVEL=Debug`.

For example,

```
TT_METAL_LOGGER_LEVEL=Debug ./build/test/tt_metal/test_add_two_ints
```

### Building and viewing the documentation locally

1. First, ensure that you have [built the project and activated the Python
environment](docs/source/get_started/get_started.rst).

2. Build the HTML Documentation page.

```
cd docs
make all
```

You can customize the port by using the `PORT=<port>` environment variable. If
you're using a customer-facing cloud machine, please disregard this point.

3. Navigate to the docs page.

Navigate your web browser to `http://<ip address>:<port>`, where `<ip address>`
is the IP address of the machine on which you launched the web server. For
example: `http://10.250.37.37:4242`, for port ``4242``.

If you forwarded your port, navigate to `http://localhost:8888`.

### Cleaning the dev environment with `make nuke`

Normally, `make clean` only clears out build artifacts. It does **not** delete
the built Python dev environment stored at `build/python_env/`.

To delete absolutely everything including the Python environment, use `make
nuke`.

## Running tests on tt-metal

Ensure you're in a developer environment with necessary environment variables
set as documentating in the [developing section](#developing-tt-metal).

This includes the environment variables, Python dev environment etc.

### Running pre/post-commit regressions

You must run regressions before you commit something.

These regressions will also run after every pushed commit to the GitHub repo.

```
make build
make tests
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type post_commit
```

If changes affect `tensor` or `tt_dnn` libraries, run this suite of pytests
which tests `tensor` APIs and `tt_dnn` ops. These are also tested in post
commit.

```
pytest tests/python_api_testing/unit_testing/ -vvv
pytest tests/python_api_testing/sweep_tests/pytests/ -vvv
```

### Running model performance tests

After building the repo and activating the dev environment with the appropriate
environment variables, you have two options for running performance regressions
on model tests.

If you are using a machine with virtual machine specs, please use

```
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type models_performance_virtual_machine
```

If you are using a machine with bare metal machine specs, please use

```
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type models_performance_bare_metal
```

## Debugging tips

- To print within a kernel the following must be added:
  - In the C++ python binding API file: `#include "tt_metal/impl/debug/dprint_server.hpp"`
  - In the same file, before launching your kernel : `    tt_start_debug_print_server(<device>, {<pci slot>}, {{<physical core coordinates>}});`
  - Note for core 0,0 it is 1,1
  - You can get the physical core given a logical core with the following call: `device->worker_core_from_logical_core(<logical_core>);`
  - In the kernel: `#include "debug/dprint.h"`
  - To print in the kernel : `DPRINT << <variable to print> << ENDL();`
- To use GDB to debug the C++ python binding itself:
  - Build with debug symbols `make build config=debug`
  - Ensure the python file you wish to debug, is standalone and has a main function
  - Run `gdb --args python <python file> `
  - You can add breakpoints for future loaded libraries

## Contribution standards

### File structure and formats

- Every source file must have the appropriate SPDX header at the top following
  the [Linux conventions](https://elixir.bootlin.com/linux/v6.5.1/source/Documentation/process/license-rules.rst#L71)
  for C++ source files, RST files, ASM files, and
  scripts. For Python files, we are to use this convention:

  ```
  # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

  # SPDX-License-Identifier: Apache-2.0
  ```

  For C++ header files, we will treat them as C++ source files and use this
  convention:

  // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
  //
  // SPDX-License-Identifier: Apache-2.0

### CI/CD Guidelines

- Revert commits on main which fail post-commit tests immediately.
- There shall be a recurring scheduled meeting concerning post-commit tests
  where:
  - Certain codeowners and project-specific members review current tests in
    post-commit.
  - Certain codeowners and project-specific members decide whether to
    remove/add any current tests in post-commit as project priorities change on
    an ongoing basis.
  - Certain codeowners and project-specific members decide if we need to change
    owners or add more as project priorities change on an ongoing basis.
  - Communication channels for these decisions and meetings shall be kept
    internal to Tenstorrent with the intent of having such discussions in the
    open later.
- Non-post-commit pipelines will not necessarily mean we have to revert the
  breaking commit, however any broken pipelines will be considered a priority
  bug fix.
- The responsibility of identifying, announcing status-tracking, and escalating
  broken non-post-commit pipelines will be the responsibility of codeowners
  whose tests are in the said non-post-commit pipeline.
  - In the case of the model performance test pipeline, there are codeowners
    for such tests. However, it is the collective responsibility of all
    developers to ensure that we do not regress this pipeline.

### Documentation

- Any API changes must be accompanied with appropriate documentation changes.

### Git rules and guidelines

- Any commit message must be accompanied with an appropriate GitHub issue
  number with a colon and following message. The message must start with an
  imperative verb and descripton of what was done. Preferably a reason is
  included. Ex.
  ```
  #41: Fix data format error in Gelu op.
  ```

- The following is not allowed in commit messages:
  - Commit messages which state that a code review or comments are being
    addressed. You must explicitly state what you are doing in each commit even
    if it's just cosmetic.

### Code reviews

- A PR must be opened for any code change with the following criteria:
  - Be approved, by a maintaining team member and any codeowners whose modules
    are relevant for the PR.
  - Pass any required post-commit pipelines rebased on the latest main. These
    pipelines will generally, but not always, be defined in
    `.github/workflows/all-post-commit-workflows.yaml`.
  - Pass any acceptance criteria mandated in the original issue.
  - Pass any testing criteria mandated by codeowners whose modules are relevant
    for the PR.
- Avoid opening/re-opening/push new commits to PRs before you're ready for
  review and start running pipelines. This is because we don't want to clog
  our pipelines with unnecessary runs that developers may know will fail
  anyways.

### New feature and design specifications

- New or changing features require the following accompanying documentation:
  - An architectural change plan approved by maintaining team members.
  - A design plan with associated GitHub project/large containing issue.
    with sub-issues for proper documentation of project slices.
  - An appropriate test plan with issues.

### Release flows

- Any release must be externally-available artifacts generated by a workflow
  on a protected branch.

### Logging, assertions, and exceptions

- Use Loguru for Python logging.
- Use Tenstorrent logger for C++ logging.

## Hardware troubleshooting

### Resetting an accelerator board

If a Tenstorrent chip seems to hang and/or is producing unexpected behaviour,
you may try a software reset of the board.

For Grayskull: `tt-smi -tr all`

For Wormhole: `tt-smi -wr all wait`

If the software reset does not work, unfortunately you will have to power cycle
the board. This usually means rebooting the host of a board.
