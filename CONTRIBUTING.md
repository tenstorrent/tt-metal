## Contribution standards
This project has adopted C++ formatting and style as defined in `.clang-format`.
There are additional requirements such as license headers.

## Pre-commit Hook Integration for Formatting and Linting

As part of maintaining consistent code formatting across the project, we have integrated the [pre-commit](https://pre-commit.com/) framework into our workflow. The pre-commit hooks will help automatically check and format code before commits are made, ensuring that we adhere to the project's coding standards.

### What is Pre-commit?

Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. It helps catch common issues early by running a set of hooks before code is committed, automating tasks like:

- Formatting code (e.g., fixing trailing whitespace, enforcing end-of-file newlines)
- Running linters (e.g., `clang-format`, `black`, `flake8`)
- Checking for merge conflicts or other common issues.

For more details on pre-commit, you can visit the [official documentation](https://pre-commit.com/).

### How to Set Up Pre-commit Locally

To set up pre-commit on your local machine, follow these steps:

1. **Install Pre-commit**:
   Ensure you have Python installed, then run:

   ```bash
   pip install pre-commit
   ```

   *Note:* pre-commit is already installed if you are using the python virtual environment.
2. **Install the Git Hook Scripts**:
   In your local repository, run the following command to install the pre-commit hooks:

   ```bash
   pre-commit install
   ```

   This command will configure your local Git to run the defined hooks automatically before each commit.
3. **Run Pre-commit Hooks Manually**:
   You can also run the hooks manually against all files at any time with:

   ```bash
   pre-commit run --all-files
   ```

### Ignoring formatting commits in git blame

In order to ignore specific commits from git blame, we need to configure git to use ignore file when executing the git blame command.

```bash
# Set the blame.ignoreRevsFile configuration in your local git config
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

This command tells git to reference the specified file for commits to ignore when you run git blame. This configuration can also be set globally or system-wide by using `--global` or `--system`.
Once you've configured git blame with the ignore-revs file, you can run git blame as you normally would:

```bash
git blame <file>
```

With the configuration set, `git blame` will skip over the commits listed in the `.git-blame-ignore-revs` file, giving you a clearer picture of who made meaningful changes to the file.
