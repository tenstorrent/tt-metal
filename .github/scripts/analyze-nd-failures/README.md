# Non-Deterministic Failure Analysis System

This system analyzes non-deterministic (ND) failures in GitHub Actions test runs and suggests code changes that could help prevent these failures from occurring in the future.

## Overview

The system consists of:

1. **Analysis Prompt** (`analysis_prompt.md`) - Detailed instructions for the AI analysis
2. **Download Script** (`download_job_logs.sh`) - Downloads logs and artifacts from GitHub Actions jobs
3. **Analysis Script** (`analyze_nd_failures.sh`) - Orchestrates the entire analysis process using Claude CLI

## Prerequisites

1. **GitHub CLI (gh)** - For downloading logs and artifacts
   ```bash
   # Install: https://cli.github.com/
   gh auth login
   ```

2. **jq** - For JSON processing
   ```bash
   # Ubuntu/Debian
   sudo apt-get install jq

   # macOS
   brew install jq
   ```

3. **Claude CLI** - For AI-powered analysis
   ```bash
   # Install: https://code.claude.com/
   # Ensure Claude CLI is set up and authenticated on your machine
   ```

4. **GitHub Authentication** - Ensure you have access to the `tenstorrent/tt-metal` repository

## Quick Start

### Setup (One-Time)

1. Install GitHub CLI and authenticate:
   ```bash
   # macOS
   brew install gh

   # Ubuntu/Debian
   sudo apt-get install gh

   gh auth login
   ```

2. Install jq:
   ```bash
   # macOS
   brew install jq

   # Ubuntu/Debian
   sudo apt-get install jq
   ```

3. Ensure Claude CLI is installed and configured (see prerequisites above)

### Basic Usage

Analyze a single failed job:

```bash
cd /tt-metal/.github/scripts/analyze-nd-failures
./analyze_nd_failures.sh https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
```

### Using Different Claude Models

The script supports multiple Claude models. Choose based on your needs:

- **sonnet** (default) - Best balance of quality and speed for most tasks
- **haiku** - Fastest and most cost-effective for simple analyses
- **sonnet-1M** - Sonnet with 1M token context window for very long prompts
- **opus** - Most intelligent model for complex reasoning (requires Pro+)

```bash
# Use default (sonnet)
./analyze_nd_failures.sh <job_url>

# Use Opus for complex analysis
./analyze_nd_failures.sh --model opus <job_url>

# Use Haiku for quick analysis
./analyze_nd_failures.sh --model haiku <job_url>

# Use Sonnet with extended context
./analyze_nd_failures.sh --model sonnet-1M <job_url>
```

## Usage

### Basic Usage

Analyze a single failed job:

```bash
./analyze_nd_failures.sh https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
```

### Multiple Jobs

Analyze multiple jobs that failed for the same reason:

```bash
./analyze_nd_failures.sh \
  https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210 \
  https://github.com/tenstorrent/tt-metal/actions/runs/1234567891/job/9876543211 \
  https://github.com/tenstorrent/tt-metal/actions/runs/1234567892/job/9876543212
```

### Using a File with URLs

Create a file `failed_jobs.txt`:

```
# Failed jobs for "device timeout" issue
https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
https://github.com/tenstorrent/tt-metal/actions/runs/1234567891/job/9876543211
```

Then run:

```bash
./analyze_nd_failures.sh --file failed_jobs.txt
```

### Advanced Options

```bash
# Specify custom output directory
./analyze_nd_failures.sh --output-dir /path/to/output <job_urls>

# Skip downloading (use existing downloaded logs)
./analyze_nd_failures.sh --skip-download

# Use specific Claude model
./analyze_nd_failures.sh --model opus <job_urls>

# Prevent overwriting existing analysis folders (append number suffix)
./analyze_nd_failures.sh --no-overwrite <job_urls>

# Combine options
./analyze_nd_failures.sh --file urls.txt --model sonnet-1M --skip-download --no-overwrite

# Create a PR with the suggested fixes after analysis
./analyze_nd_failures.sh --create-pr <job_url>

# Create PR against a different base branch
./analyze_nd_failures.sh --create-pr --pr-base develop <job_url>
```

### Creating PRs from Analysis

The `--create-pr` flag automates the entire fix workflow:
1. Analyzes the failure (as normal)
2. Uses Claude to implement the primary recommended fix
3. Creates a branch, commits changes, and opens a PR

```bash
# Full workflow: analyze and create PR
./analyze_nd_failures.sh --create-pr <job_url>

# Use a more capable model for complex fixes
./analyze_nd_failures.sh --create-pr --model opus <job_url>
```

You can also create a PR from an existing analysis using the standalone script:

```bash
# Create PR from existing analysis
./create_pr_from_analysis.sh path/to/analysis_result.md

# Dry run - make changes but don't push/create PR
./create_pr_from_analysis.sh --dry-run path/to/analysis_result.md

# Specify base branch and model
./create_pr_from_analysis.sh --base develop --model opus path/to/analysis_result.md
```

#### Preventing Overwrites

By default, running the same analysis twice will overwrite the previous results in the same folder. Use `--no-overwrite` to preserve previous analyses:

- **Default behavior**: `singlecarddemotestsvitn300func_device_timeout/` gets overwritten
- **With `--no-overwrite`**: Creates `singlecarddemotestsvitn300func_device_timeout_1/`, `_2/`, etc.

This is useful when you want to:
- Compare analyses from different time periods
- Keep multiple analysis runs for the same failure
- Track how analysis quality changes over time

## URL Formats

The script accepts GitHub Actions URLs in these formats:

1. **Full job URL** (preferred):
   ```
   https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/job/<job_id>
   ```

2. **Run URL** (downloads all jobs in the run):
   ```
   https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>
   ```

3. **With attempt number**:
   ```
   https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/attempts/<attempt>/job/<job_id>
   ```

## How It Works

1. **Download Phase**:
   - Extracts run_id and job_id from URLs
   - Downloads job logs, annotations, and artifacts using GitHub CLI
   - Organizes downloads in run-specific directories

2. **Context Preparation**:
   - Extracts job name and error type from logs
   - Creates a unique directory name: `<job_name>_<error_abbrev>`
   - Organizes logs and prepares analysis context

3. **Analysis Phase**:
   - Combines the analysis prompt with log excerpts and error messages
   - Sends everything to Claude CLI for analysis using the selected model

4. **Output**:
   - Individual analysis results for each job
   - Combined analysis if multiple jobs were analyzed
   - Detailed recommendations with code change suggestions

## Output Structure

Results are organized in `build_ND_analysis/` at the repository root:

```
build_ND_analysis/
├── <job_name>_<error_abbrev>/
│   ├── downloaded_logs/
│   │   ├── logs/                    # Job logs
│   │   └── artifacts/               # Job artifacts
│   └── analysis_output/
│       ├── context/
│       │   └── logs/                # Logs copied for analysis
│       ├── full_prompt.md           # Complete prompt sent to Claude
│       └── analysis_result.md       # AI analysis results
└── combined_analysis.md              # Combined results (if multiple jobs)
```

Each run creates a separate folder, so you can analyze multiple failures without overwriting previous results.

## Understanding the Analysis Results

The analysis results follow this structure:

```markdown
## Failure Summary
Brief description of what failed and when

## Root Cause Analysis
Detailed analysis of why the failure occurred

## Fixability Assessment
Whether this can be fixed in tt-metal code

## Recommended Code Changes

### Priority 1: [High Impact, High Feasibility]
Specific code changes with file paths

### Priority 2: [Medium Priority]
Additional suggestions

### Priority 3: [Lower Priority]
Other improvements

## Implementation Notes
Important considerations for implementation
```

## Common Non-Deterministic Failures

The system is designed to analyze failures like:

- **Device Initialization Failures**: "failed to initialize chip"
- **Timeout Failures**: "device timed out in X thread"
- **Connection Issues**: "Physical Discovery found missing channel/port connections"
- **Hardware Errors**: "Connection mismatch detected"
- **Resource Exhaustion**: Out of memory, handles, etc.
- **Race Conditions**: Timing-dependent failures

## Troubleshooting

### Claude CLI Not Found

If you get an error about Claude CLI:

```bash
# Verify Claude CLI is installed
which claude

# Check Claude CLI documentation for installation:
# https://code.claude.com/
```

### GitHub Authentication Issues

```bash
# Check authentication status
gh auth status

# Login if needed
gh auth login

# Refresh token if expired
gh auth refresh
```

### Download Failures

If downloads fail:

1. Check your GitHub token permissions
2. Verify the job URLs are correct
3. Ensure you have access to the repository
4. Check network connectivity

### Analysis Failures

If Claude analysis fails:

1. Verify Claude CLI is installed and configured
2. Check that the prompt file exists: `analysis_prompt.md`
3. Ensure logs were downloaded successfully
4. Try running with `--skip-download` if logs already exist
5. Try a different model (e.g., `--model haiku` for faster processing)

### Invalid Model Error

If you see an invalid model error:

```bash
# Valid models are: haiku, sonnet, sonnet-1M, opus
./analyze_nd_failures.sh --model sonnet <job_url>
```

## Limitations

1. **Hardware Issues**: Some failures require firmware or hardware driver changes that are outside tt-metal's scope
2. **External Dependencies**: Failures due to GitHub Actions infrastructure or external services cannot be fixed in code
3. **Rare Failures**: Very rare failures may not have enough data for meaningful analysis
4. **Claude Dependency**: Requires Claude CLI to be installed and configured

## Examples

### Example 1: Single Job Analysis

```bash
./analyze_nd_failures.sh \
  https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
```

### Example 2: Batch Analysis with Opus Model

Create `device_timeout_failures.txt`:

```
# Device timeout failures from last week
https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
https://github.com/tenstorrent/tt-metal/actions/runs/1234567891/job/9876543211
https://github.com/tenstorrent/tt-metal/actions/runs/1234567892/job/9876543212
```

Run:

```bash
./analyze_nd_failures.sh --file device_timeout_failures.txt --model opus
```

### Example 3: Re-analyze Existing Logs

If you've already downloaded logs:

```bash
./analyze_nd_failures.sh --skip-download --model sonnet-1M
```

### Example 4: Quick Analysis with Haiku

For a quick analysis of a simple failure:

```bash
./analyze_nd_failures.sh --model haiku <job_url>
```

### Example 5: Preserve Multiple Analysis Runs

To keep multiple analysis runs for the same failure without overwriting:

```bash
# First run creates: singlecarddemotestsvitn300func_device_timeout/
./analyze_nd_failures.sh --no-overwrite <job_url>

# Second run creates: singlecarddemotestsvitn300func_device_timeout_1/
./analyze_nd_failures.sh --no-overwrite <job_url>

# Third run creates: singlecarddemotestsvitn300func_device_timeout_2/
./analyze_nd_failures.sh --no-overwrite <job_url>
```

## Model Selection Guide

Choose the right Claude model for your analysis:

- **haiku**: Use for quick, simple analyses. Fastest and most cost-effective.
- **sonnet** (default): Best for most use cases. Good balance of quality, speed, and cost.
- **sonnet-1M**: Use when you have very long prompts or need extended context. Same quality as sonnet but with 1M token window.
- **opus**: Use for complex, nuanced failures that require deep reasoning. Most capable but slower and more expensive.

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the script output for error messages
3. Verify all prerequisites are installed
4. Check GitHub Actions job URLs are accessible
5. Review the script help: `./analyze_nd_failures.sh --help`

## Contributing

When improving this system:

1. Update `analysis_prompt.md` if the analysis requirements change
2. Test with real failure cases
3. Document any new features in this README
4. Ensure scripts handle edge cases gracefully

## License

This system is part of the tt-metal repository and follows the same license terms.
