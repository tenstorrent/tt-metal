# Quick Start Guide

## Setup (One-Time)

1. **Install GitHub CLI**:
   ```bash
   # macOS
   brew install gh

   # Ubuntu/Debian
   sudo apt-get install gh
   ```

2. **Authenticate GitHub CLI**:
   ```bash
   gh auth login
   ```

3. **Install jq**:
   ```bash
   # macOS
   brew install jq

   # Ubuntu/Debian
   sudo apt-get install jq
   ```

4. **Install GitHub Copilot CLI**:
   ```bash
   npm install -g @githubnext/github-copilot-cli
   github-copilot-cli auth
   ```

## Basic Usage

### Analyze a Single Failed Job

1. Get the job URL from GitHub Actions (e.g., `https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210`)

2. Run the analysis:
   ```bash
   cd /tt-metal/.github/scripts/analyze-nd-failures
   ./analyze_nd_failures.sh <job_url>
   ```

3. Review the results in `analysis_output/`

### Analyze Multiple Jobs

Create a file `failed_jobs.txt`:
```
https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210
https://github.com/tenstorrent/tt-metal/actions/runs/1234567891/job/9876543211
```

Run:
```bash
./analyze_nd_failures.sh --file failed_jobs.txt
```

## What Happens

1. **Downloads logs** from GitHub Actions
2. **Extracts context** (test files, source files mentioned in errors)
3. **Creates analysis prompt** with all relevant information
4. **Runs Copilot analysis** to get recommendations
5. **Saves results** in `analysis_output/`

## Output Files

- `analysis_output/context_*/analysis_result.md` - AI analysis with recommendations
- `analysis_output/context_*/full_prompt.md` - Complete prompt sent to Copilot
- `analysis_output/context_*/logs/` - Downloaded job logs
- `analysis_output/combined_analysis.md` - Combined results (if multiple jobs)

## Troubleshooting

### "Copilot CLI not found"
```bash
npm install -g @githubnext/github-copilot-cli
github-copilot-cli auth
```

### "GitHub authentication failed"
```bash
gh auth login
gh auth refresh
```

### "Analysis failed"
Check the manual analysis script created in `analysis_output/` - you can run Copilot manually using the instructions there.

## Next Steps

1. Review the analysis results
2. Implement recommended code changes
3. Test the changes
4. Monitor if failures decrease

For more details, see [README.md](README.md).
