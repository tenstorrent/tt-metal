# Non-Deterministic Failure Analysis

Analyzes GitHub Actions test failures using Claude to identify root causes and suggest fixes.

## Quick Start

```bash
# Analyze a failed job
./analyze_nd_failures.sh <job_url>

# With a custom folder name (recommended)
./analyze_nd_failures.sh --name "vit-timeout" <job_url>

# Full workflow: analyze and create a PR
./analyze_nd_failures.sh --create-pr --model opus <job_url>
```

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| `gh` | GitHub CLI | `brew install gh` or `apt install gh` |
| `jq` | JSON parsing | `brew install jq` or `apt install jq` |
| `claude` | AI analysis | [claude.ai/code](https://claude.ai/code) |

After installing, authenticate with GitHub:
```bash
gh auth login
```

## Usage

### Basic Commands

```bash
# Single job
./analyze_nd_failures.sh https://github.com/tenstorrent/tt-metal/actions/runs/123/job/456

# Multiple jobs
./analyze_nd_failures.sh <url1> <url2> <url3>

# From file (one URL per line)
./analyze_nd_failures.sh --file failed_jobs.txt
```

### Options

| Option | Description |
|--------|-------------|
| `--name, -n <name>` | Human-readable folder name (e.g., `--name "vit-demo-timeout"`) |
| `--model, -m <model>` | Claude model: `haiku`, `sonnet` (default), `sonnet-1M`, `opus` |
| `--output-dir, -o <dir>` | Output directory (default: `build_nd_analysis/`) |
| `--skip-download` | Re-analyze existing logs |
| `--no-overwrite` | Create numbered folders instead of overwriting |
| `--create-pr` | Auto-create a PR with suggested fixes |
| `--pr-base <branch>` | Base branch for PR (default: `main`) |

### Examples

```bash
# Quick analysis with Haiku
./analyze_nd_failures.sh --model haiku <url>

# Deep analysis with Opus and custom name
./analyze_nd_failures.sh --name "matmul-hang" --model opus <url>

# Batch analysis
./analyze_nd_failures.sh --file timeout_failures.txt --model sonnet-1M

# Full automation: analyze + create PR
./analyze_nd_failures.sh --create-pr --model opus <url>
```

## Output Structure

Results are saved to `build_nd_analysis/` in the repository root:

```
build_nd_analysis/
└── <name>--<error-type>/          # e.g., "vit-demo--device-timeout"
    ├── downloaded_logs/
    │   ├── logs/                  # Job logs
    │   └── artifacts/             # Test artifacts
    └── analysis_output/
        ├── context/logs/          # Logs for analysis
        ├── full_prompt.md         # Prompt sent to Claude
        └── analysis_result.md     # Analysis results
```

### Folder Naming

- **With `--name`**: Uses your name directly (e.g., `--name "vit-issue"` → `vit-issue/`)
- **Without `--name`**: Auto-generates from job name and error type (e.g., `demo-tests-wormhole--device-timeout/`)

## Creating PRs

### Automatic (recommended)
```bash
./analyze_nd_failures.sh --create-pr <url>
```

### From existing analysis
```bash
./create_pr_from_analysis.sh path/to/analysis_result.md
./create_pr_from_analysis.sh --dry-run path/to/analysis_result.md  # Preview only
```

## Model Selection

| Model | Best For |
|-------|----------|
| `haiku` | Quick triage, simple failures |
| `sonnet` | Most analyses (default, good balance) |
| `sonnet-1M` | Very long logs (1M token context) |
| `opus` | Complex failures, deep reasoning |

## Supported URL Formats

```
# Full job URL (preferred)
https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/job/<job_id>

# Run URL (downloads all jobs)
https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>

# With attempt number
https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/attempts/<n>/job/<job_id>
```

## Troubleshooting

### "GitHub CLI not authenticated"
```bash
gh auth login
gh auth status  # Verify
```

### "Claude CLI not found"
Install from [claude.ai/code](https://claude.ai/code) and ensure it's in your PATH.

### Analysis times out
- Try `--model haiku` for faster processing
- Check if logs are unusually large

### No changes made by Claude
- The analysis might not contain actionable code changes
- Try re-running with `--model opus` for better reasoning

## Files

| File | Purpose |
|------|---------|
| `analyze_nd_failures.sh` | Main analysis script |
| `create_pr_from_analysis.sh` | Creates PRs from analysis |
| `download_job_logs.sh` | Downloads GitHub Actions logs |
| `extract_job_info.sh` | Parses job URLs |
| `analysis_prompt.md` | Instructions for Claude |
