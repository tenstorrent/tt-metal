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
   - Do **not merge** the change into the submodule’s main branch until:
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
