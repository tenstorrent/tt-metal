# lint_clang_format.sh

## Overview

This bash script facilitates the process of applying `clang-format` across the repository. Users
can select to format either all files within the repository (`all` mode) or only those files that
have been modified (`diff` mode) compared to the main branch or a specified base commit.

## Usage

```bash
./lint_clang_format.sh [-m <diff|all>] [-n] [-f]
```

Options:

- `-m <diff|all>`: Choose to format all relevant C++ files in the repository (all) or only those
  modified on your branch (diff).
- `-n`: Perform a dry run, listing files that would be formatted without actually running
  `clang-format`.
- `-f`: Automatically fix formatting issues.
