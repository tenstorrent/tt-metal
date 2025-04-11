# PR Reviewer Tool - Review Optimizer

A tool to automatically select reviewers for pull requests based on code ownership.

# Codeowner Installation

Before installing this package, ensure you have the `codeowners` CLI installed:

## macOS
```sh
brew tap hmarr/tap
brew install codeowners
```

## Other platforms
```sh
go install github.com/hmarr/codeowners/cmd/codeowners@latest
```

## Review Optimizer Installation

### For Development (Editable Install)

```bash
cd /path/to/tt-metal/cli_tool
pip install -e .
```

### For Regular Use

```bash
cd /path/to/tt-metal/cli_tool
pip install .
```

## Usage

After installation, you can use the tool from anywhere with the following command syntax:
```bash
get_reviewers PR_NUMBER -include name1 name2 ... -skip name3 name4 ...
```

```bash
get_reviewers PR_NUMBER
```


## Features

- Automatically selects reviewers based on code ownership
- Optimizes reviewer selection to minimize the number of people needed
- Integrates with GitHub CLI for PR information