# Non-Deterministic Failure Analysis System

This system analyzes non-deterministic (ND) failures in GitHub Actions test runs and suggests code changes that could help prevent these failures from occurring in the future.

## Overview

The system consists of:

1. **Analysis Prompt** (`analysis_prompt.md`) - Detailed instructions for the AI analysis
2. **Download Script** (`download_job_logs.sh`) - Downloads logs and artifacts from GitHub Actions jobs
3. **Analysis Script** (`analyze_nd_failures.sh`) - Orchestrates the entire analysis process using GitHub Copilot CLI

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

3. **GitHub Copilot CLI** - For AI-powered analysis
   ```bash
   npm install -g @githubnext/github-copilot-cli
   github-copilot-cli auth
   ```

4. **GitHub Authentication** - Ensure you have access to the `tenstorrent/tt-metal` repository

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
```

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
   - Organizes downloads in `downloaded_logs/` directory

2. **Context Preparation**:
   - Analyzes logs to identify relevant test files
   - Extracts source files mentioned in error messages
   - Creates a structured context directory with all relevant files

3. **Analysis Phase**:
   - Combines the analysis prompt with log excerpts and error messages
   - Includes relevant test and source files in the context
   - Sends everything to GitHub Copilot CLI for analysis

4. **Output**:
   - Individual analysis results for each job in `analysis_output/`
   - Combined analysis if multiple jobs were analyzed
   - Detailed recommendations with code change suggestions

## Output Structure

```
analysis_output/
├── context_run_1234567890_attempt_1/
│   ├── logs/                    # Job logs
│   ├── test_files/              # Relevant test files
│   ├── source_files/            # Relevant source files
│   ├── summary.txt              # Context summary
│   ├── full_prompt.md           # Complete prompt sent to Copilot
│   └── analysis_result.md       # AI analysis results
└── combined_analysis.md         # Combined results (if multiple jobs)
```

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

### Copilot CLI Not Found

If you get an error about Copilot CLI:

```bash
# Install globally
npm install -g @githubnext/github-copilot-cli

# Authenticate
github-copilot-cli auth

# Verify installation
github-copilot-cli --version
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

If Copilot analysis fails:

1. Verify Copilot CLI is installed and authenticated
2. Check that the prompt file exists: `analysis_prompt.md`
3. Ensure logs were downloaded successfully
4. Try running with `--skip-download` if logs already exist

## Limitations

1. **Hardware Issues**: Some failures require firmware or hardware driver changes that are outside tt-metal's scope
2. **External Dependencies**: Failures due to GitHub Actions infrastructure or external services cannot be fixed in code
3. **Rare Failures**: Very rare failures may not have enough data for meaningful analysis
4. **Copilot Dependency**: Requires GitHub Copilot CLI to be installed and configured

## Future Enhancements

Potential improvements:

- Support for analyzing failures from multiple repositories
- Integration with GitHub Actions workflows
- Automatic retry suggestions based on analysis
- Historical trend analysis
- Integration with issue tracking systems

## Contributing

When improving this system:

1. Update `analysis_prompt.md` if the analysis requirements change
2. Test with real failure cases
3. Document any new features in this README
4. Ensure scripts handle edge cases gracefully

## Examples

### Example 1: Single Job Analysis

```bash
./analyze_nd_failures.sh \
  https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
```

### Example 2: Batch Analysis

Create `device_timeout_failures.txt`:

```
# Device timeout failures from last week
https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
https://github.com/tenstorrent/tt-metal/actions/runs/1234567891/job/9876543211
https://github.com/tenstorrent/tt-metal/actions/runs/1234567892/job/9876543212
```

Run:

```bash
./analyze_nd_failures.sh --file device_timeout_failures.txt
```

### Example 3: Re-analyze Existing Logs

If you've already downloaded logs:

```bash
./analyze_nd_failures.sh --skip-download
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the script output for error messages
3. Verify all prerequisites are installed
4. Check GitHub Actions job URLs are accessible

## License

This system is part of the tt-metal repository and follows the same license terms.
