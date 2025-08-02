# 📘 Rules of Engagement: Submodule Repository Changes

This repository (`tt-llk`) is **consumed by another repository** (`tt-metal`) as a git submodule. The following rules aim to ensure safe and coordinated development across both projects.

## ✅ General Principles

- This repository is **used as a git submodule** by `tt-metal`.
- **Prefer backward compatibility** when possible, but breaking changes are acceptable when properly coordinated.
- **Breaking changes must be coordinated** with `tt-metal` maintainers before being merged here.

## 🔒 Breaking Changes

### What Is a Breaking Change?

A breaking change is any change that:

- Removes or alters public APIs, interfaces, or headers.
- Changes existing behavior in a non-backward-compatible way.
- Adds new build/runtime dependencies.
- Modifies file structure or exported symbols relied on by the `tt-metal` repo.

### Breaking Change Workflow

1. **Localize the impact**
   - Favor additive over destructive changes.
   - If a destructive change is necessary, introduce it in a way that requires minimal modifications to the existing code. If larger or additional destructive changes are needed, split them into separate pull requests.

2. **Communicate early**
   - Notify maintainers from the `tt-metal` repo.
   - Provide context: what will break, why the change is needed, and provide an estimated timeline.
   - Allow time for feedback and coordination before implementing changes.

3. **Test with the `tt-metal` repo first**
   - Open a **pull request in the tt-metal** that:
     - Points to your branch in the submodule.
     - Adapts `tt-metal` codebase to the breaking change.
     - Passes **all CI checks** using that branch:
         - [All post-commit checks](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml)
         - [Blackhole post-commit checks](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml)
         - If your change is affecting ttnn operations, additional checks might be required by the maintainers.

4. **Merge only after validation**
   - Do **not merge** the change into the submodule's main branch until:
     - The `tt-metal` PR is **approved**.
     - **All checks are passing**.
     - Reviewers from both repositories have approved your changes.

## 🔁 Submodule Update Guidelines

- Never update the submodule pointer in the `tt-metal` to an **untested** or **unstable** commit.
- Submodule updates must reference a **compatible and validated** commit on `main` branch.

## 🚫 Anti-Patterns

Avoid the following:

- Merging breaking changes to the submodule **before verifying compatibility**.
- Leaving the submodule pointer in a detached or unverified state.
- Introducing broad interface changes without notifying the maintainers.

## 📣 Summary

| Step        | Action                                                                |
|-------------|-----------------------------------------------------------------------|
| 🔍 Plan     | Discuss breaking changes and tag downstream maintainers               |
| 🧪 Test     | Open a PR in the tt-metal repo using your submodule branch            |
| ✅ Validate | Ensure CI is green and the PR is approved in the tt-metal repo        |
| 🔀 Merge    | Only merge to submodule main after successful validation and approval |

## 🧪 Running tt-metal Integration Tests

You can run integration tests in the `tt-metal` repository directly from your `tt-llk` pull request using GitHub labels. This is useful for testing your changes against the full `tt-metal` test suite without manually creating a PR in the `tt-metal` repository.

### Available Test Labels

Add one or more of these labels to your pull request to trigger the corresponding tests:

- **`metal-post-commit-tests`** - **Smart detection** - Automatically determines which tests to run based on your file changes:
  - If you modify files in `tt_llk_blackhole/`, runs Blackhole integration tests
  - If you modify files in `tt_llk_wormhole_b0/`, runs Wormhole integration tests
  - If you modify both directories, runs both test suites
  - If no relevant changes detected, skips testing

- **`wormhole-integration-tests`** - **Targeted testing** - Always runs Wormhole LLK unit tests using your branch changes, regardless of which files were modified

- **`blackhole-integration-tests`** - **Targeted testing** - Always runs Blackhole LLK unit tests using your branch changes, regardless of which files were modified

### How It Works

1. **Add a label** to your pull request (see labels above)
2. **Automatic branch handling**:
   - For **fork PRs**: The system automatically mirrors your branch to the main repository
   - For **internal PRs**: Uses your branch directly
3. **Test execution**: Triggers the appropriate test suite in the `tt-metal` repository
4. **Results**: You'll receive a comment on your PR with:
   - Test configuration details
   - Direct link to the running tests
   - Status updates

### When to Use Post-Commit Tests

- **Before merging breaking changes** - Validate compatibility with `tt-metal`
- **Testing new features** - Ensure your changes work in the full system
- **Debugging issues** - Reproduce problems in the complete environment

### Manual Testing Alternative

For more control, you can still manually:

1. Create a PR in `tt-metal` that points to your `tt-llk` branch
2. Run the desired test workflows manually
3. Review detailed results and logs

The automated integration tests are a convenience feature - manual testing in `tt-metal` remains the authoritative validation method for breaking changes.

---

By following this process, we ensure safe evolution of shared code while minimizing disruption for downstream projects.

---

# 🛠️ Development Standards

Beyond submodule coordination, this repository follows standard development practices for code quality and consistency.

## 🤝 Code of Conduct

This project and everyone participating in it is expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the standards of behavior we expect from all contributors and community members.

## Code Formatting and Style

This project has adopted C++ formatting and style as defined in `.clang-format`.
There are additional requirements such as license headers.

## Pre-commit Hook Integration for Formatting and Linting

As part of maintaining consistent code formatting across the project, we have integrated the [pre-commit](https://pre-commit.com/) framework into our workflow. The pre-commit hooks will help automatically check and format code before commits are made, ensuring that we adhere to the project's coding standards.

### What is Pre-commit?

Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. It helps catch common issues early by running a set of hooks before code is committed, automating tasks like:

- Formatting code (e.g., fixing trailing whitespace, enforcing end-of-file newlines)
- Running linters (e.g., `clang-format`, `black`, `pylint`)
- Fixing common spelling mistakes.
- Checking for merge conflicts or other common issues.

For more details on pre-commit, you can visit the [official documentation](https://pre-commit.com/).

### How to Set Up Pre-commit Locally

Pre-commit hooks are **automatically installed** when you run the testing environment setup:

```bash
cd tests
./setup_testing_env.sh
```

This script will install pre-commit and configure the git hooks for you.

#### Manual Setup (Alternative)

If you prefer to set up pre-commit manually or need to reinstall it, follow these steps:

1. **Install Pre-commit**:
   Ensure you have Python installed, then run:

   ```bash
   pip install pre-commit
   ```

   *Note:* pre-commit is already installed if you are using the Python virtual environment.
2. **Install the Git Hook Scripts**:
   In your local repository, run the following command to install the pre-commit hooks:

   ```bash
   pre-commit install
   ```

   This command will configure your local Git to run the defined hooks automatically before each commit.

3. **Commit Your Code**:
   When you run `git commit`, the pre-commit hooks will automatically execute. If any issues are found, the commit will be aborted, and you will see a message indicating what needs to be fixed. Simply correct the issues and re-commit.

   If a hook makes changes to your files (e.g., auto-formatting), you will need to `git add` the modified files before committing again.

4. **Run Pre-commit Hooks Manually (optional)**:
   You can also run the hooks manually against all files at any time with:

   ```bash
   pre-commit run --all-files
   ```

### `clangd` setup for IDE Integration

This project uses `clangd` to provide language server features like code completion, navigation, and live error checking. Follow these steps to set it up correctly:

1. **Install Clangd Extension**:
    First, install the `clangd` extension for your code editor (e.g., VSCode, Cursor).

2. **Download Dependencies**:
    `clangd` requires an external toolchain and several headers from the `tt-metal` project. You can download all the necessary dependencies by running the following script:

    ```bash
    cd tests
    ./setup_testing_env.sh
    cd ..
    ```

    This will download the SFPI toolchain. Headers will be downloaded automatically once you run the tests.

3. **Generate Compilation Flags**:
    With the dependencies in place, run the `setup_clangd.sh` script from the repository root. This creates the `compile_flags.txt` file that `clangd` needs. You must specify a target architecture:

    ```bash
    ./setup_clangd.sh blackhole
    ```

4. **Reload file**:
    After completing the setup, you might have to reload the file in the editor. That is not always the case.

### Spell Checking with `codespell`

We use `codespell` to automatically fix common spelling errors in our codebase and documentation. It runs on every commit and also checks commit messages for typos.

If `codespell` reports a false positive (a word that is correctly spelled but not in its dictionary), you can add it to our project's ignore list.

**How to Add a Word to the Ignore List:**

1. **Add the word**: The ignore list is located in the `.codespellignore` file in the root of the repository. Add the word you want to ignore to a new line in the file.

2. **Commit the change**: Commit the updated `.codespellignore` file along with your other changes.

### Ignoring formatting commits in git blame

In order to ignore specific commits from git blame, we need to configure git to use an ignore file when executing the git blame command.

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
