<!-- toc -->

   * [Contributing to tt-metal](#contributing-to-tt-metal)
   * [Machine setup](#machine-setup)
      * [Hugepages setup](#hugepages-setup)
   * [Developing tt-metal](#developing-tt-metal)
      * [Building and viewing the documentation locally](#building-and-viewing-the-documentation-locally)
   * [Debugging tips](#debugging-tips)
   * [Contribution standards](#contribution-standards)
      * [CI/CD Guidelines](#cicd-guidelines)
      * [Documentation](#documentation)
      * [Git rules and guidelines](#git-rules-and-guidelines)
      * [Code reviews](#code-reviews)
      * [New feature and design specifications](#new-feature-and-design-specifications)
      * [Release flows](#release-flows)
      * [Logging, assertions, and exceptions](#logging-assertions-and-exceptions)

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
  - Your PR must be approved by appropriate reviewers. We do not accept PRs
    from forked repositories.

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

## Debugging tips

- To print within a kernel the following must be added:
  - In the C++ python binding API file: `#include "tt_metal/llrt/tt_debug_print_server.hpp"`
  - In the same file, before launching your kernel : `    tt_start_debug_print_server(<device>->cluster(), {<pci slot>}, {{<physical core coordinates>}});`
  - Note for core 0,0 it is 1,1
  - You can get the physical core given a logical core with the following call: `device->worker_core_from_logical_core(<logical_core>);`
  - In the kernel: `#include "debug_print.h"`
  - To print in the kernel : `DPRINT << <variable to print> << ENDL();`
- To use GDB to debug the C++ python binding itself:
  - Build with debug symbols `make build config=debug`
  - Ensure the python file you wish to debug, is standalone and has a main function
  - Run `gdb --args python <python file> `
  - You can add breakpoints for future loaded libraries

## Contribution standards

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

### Code reviews

- A PR must be opened for any code change with the following criteria:
  - Be approved, by a maintaining team member and any codeowners whose modules
    are relevant for the PR.
  - Pass post-commit tests.
  - Pass any acceptance criteria mandated in the original issue.
  - Pass any testing criteria mandated by codeowners whose modules are relevant
    for the PR.

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
