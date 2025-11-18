#!/bin/bash

# Script to find the last successful and first failing run of a specific subjob
# Usage: ./find_boundaries.sh <workflow_name> <subjob_name>
# Example: ./find_boundaries.sh single-card-demo-tests yolov5x-N150-func

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <workflow_name> <subjob_name>"
    echo ""
    echo "Examples:"
    echo "  $0 single-card-demo-tests yolov5x-N150-func"
    echo "  $0 single-card-demo-tests vanilla_unet-N150-func"
    echo ""
    echo "The workflow_name should match the workflow file name (without .yaml extension)"
    exit 1
fi

WORKFLOW_NAME="$1"
SUBJOB_NAME="$2"

REPO="tenstorrent/tt-metal"
BASE_URL="https://github.com/${REPO}"

echo -e "${BLUE}Searching for workflow: ${GREEN}${WORKFLOW_NAME}${NC}"
echo -e "${BLUE}Looking for subjob: ${GREEN}${SUBJOB_NAME}${NC}"
echo ""

# First, find the workflow ID
echo "Finding workflow ID..."
WORKFLOW_FILE="${WORKFLOW_NAME}.yaml"
WORKFLOW_ID=$(gh api "repos/${REPO}/actions/workflows/${WORKFLOW_FILE}" --jq '.id' 2>/dev/null || echo "")

if [ -z "$WORKFLOW_ID" ]; then
    echo -e "${RED}Error: Could not find workflow '${WORKFLOW_NAME}'${NC}"
    echo "Make sure the workflow file exists at: .github/workflows/${WORKFLOW_FILE}"
    exit 1
fi

echo -e "${GREEN}Found workflow ID: ${WORKFLOW_ID}${NC}"
echo ""

# Fetch workflow runs (limit to recent runs for performance, only main branch)
echo "Fetching workflow runs from main branch..."
MAX_RUNS=200
RUNS_JSON=$(gh api "repos/${REPO}/actions/workflows/${WORKFLOW_ID}/runs?branch=main&per_page=100&page=1" --jq ".workflow_runs[0:${MAX_RUNS}]" || echo "[]")

if [ "$RUNS_JSON" = "[]" ] || [ -z "$RUNS_JSON" ]; then
    echo -e "${RED}Error: Could not fetch workflow runs${NC}"
    exit 1
fi

RUN_COUNT=$(echo "$RUNS_JSON" | jq 'length')
echo -e "${GREEN}Found ${RUN_COUNT} workflow runs${NC}"
echo ""

# Process runs to find subjob status
echo "Analyzing runs for subjob '${SUBJOB_NAME}'..."
echo ""

LAST_SUCCESSFUL_RUN=""
LAST_SUCCESSFUL_RUN_ID=""
LAST_SUCCESSFUL_COMMIT=""
FIRST_FAILING_RUN=""
FIRST_FAILING_RUN_ID=""
FIRST_FAILING_COMMIT=""

# Filter runs to only main branch, completed, non-cancelled
echo "Filtering runs (main branch, completed, non-cancelled)..."
VALID_RUNS=$(echo "$RUNS_JSON" | jq -r "[.[] | select(.head_branch == \"main\" and .status == \"completed\" and .conclusion != \"cancelled\")]")
VALID_RUN_COUNT=$(echo "$VALID_RUNS" | jq 'length')
echo -e "${GREEN}Found ${VALID_RUN_COUNT} valid runs to check${NC}"
echo ""

# Extract run IDs (newest first)
RUN_IDS=$(echo "$VALID_RUNS" | jq -r '.[].id' | head -100)

PROCESSED=0
FOUND_SUCCESS=false
FOUND_FAILURE=false

# Process runs in reverse chronological order (newest first)
# Fetch jobs on-demand and stop early when we have both results
for RUN_ID in $RUN_IDS; do
    RUN_DATA=$(echo "$VALID_RUNS" | jq -r ".[] | select(.id == ${RUN_ID})")
    RUN_COMMIT=$(echo "$RUN_DATA" | jq -r '.head_sha')
    RUN_URL="${BASE_URL}/actions/runs/${RUN_ID}"

    PROCESSED=$((PROCESSED + 1))
    echo -n "[${PROCESSED}] Checking run ${RUN_ID}... "

    # Fetch jobs page by page, stopping once we find the subjob
    # Extract only the fields we need: name, conclusion, status, id
    PAGE=1
    PER_PAGE=100
    FOUND_JOB=false

    while [ "$FOUND_JOB" = false ]; do
        # Fetch one page of jobs, extracting only needed fields
        PAGE_JOBS=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}/jobs?per_page=${PER_PAGE}&page=${PAGE}" --jq '[.jobs[] | {name, conclusion, status, id}]' 2>/dev/null || echo "[]")

        if [ "$PAGE_JOBS" = "[]" ] || [ -z "$PAGE_JOBS" ]; then
            break
        fi

        # Check if we have any jobs on this page
        JOB_COUNT=$(echo "$PAGE_JOBS" | jq 'length' 2>/dev/null || echo "0")
        if [ "$JOB_COUNT" -eq 0 ]; then
            break
        fi

        # Find the specific subjob on this page
        SUBJOB=$(echo "$PAGE_JOBS" | jq -r ".[] | select(.name == \"${SUBJOB_NAME}\" or .name == \"single-card-demo-tests / ${SUBJOB_NAME}\" or .name == \"${WORKFLOW_NAME} / ${SUBJOB_NAME}\" or (.name | endswith(\"${SUBJOB_NAME}\")) or (.name | contains(\"${SUBJOB_NAME}\")))" || echo "")

        if [ -n "$SUBJOB" ]; then
            FOUND_JOB=true
            break
        fi

        # If we got fewer jobs than per_page, we're on the last page
        if [ "$JOB_COUNT" -lt "$PER_PAGE" ]; then
            break
        fi

        PAGE=$((PAGE + 1))
    done

    if [ "$FOUND_JOB" = false ] || [ -z "$SUBJOB" ]; then
        echo -e "${YELLOW}Subjob not found${NC}"
        continue
    fi

    JOB_CONCLUSION=$(echo "$SUBJOB" | jq -r '.conclusion // "null"')
    JOB_STATUS=$(echo "$SUBJOB" | jq -r '.status')
    JOB_ID=$(echo "$SUBJOB" | jq -r '.id')
    JOB_URL="${BASE_URL}/actions/runs/${RUN_ID}/job/${JOB_ID}"

    # Check if job completed
    if [ "$JOB_STATUS" != "completed" ]; then
        echo -e "${YELLOW}Job not completed${NC}"
        continue
    fi

    if [ "$JOB_CONCLUSION" = "success" ]; then
        if [ "$FOUND_SUCCESS" = false ]; then
            LAST_SUCCESSFUL_RUN="$RUN_URL"
            LAST_SUCCESSFUL_RUN_ID="$RUN_ID"
            LAST_SUCCESSFUL_COMMIT="$RUN_COMMIT"
            FOUND_SUCCESS=true
            echo -e "${GREEN}✓ SUCCESS${NC}"

            # If we already found a failure, we can stop
            if [ "$FOUND_FAILURE" = true ]; then
                echo ""
                echo -e "${GREEN}Found both success and failure - stopping search${NC}"
                break
            fi
        else
            echo -e "${GREEN}Success (already found)${NC}"
        fi
    elif [ "$JOB_CONCLUSION" = "failure" ]; then
        if [ "$FOUND_FAILURE" = false ]; then
            FIRST_FAILING_RUN="$RUN_URL"
            FIRST_FAILING_RUN_ID="$RUN_ID"
            FIRST_FAILING_COMMIT="$RUN_COMMIT"
            FOUND_FAILURE=true
            echo -e "${RED}✗ FAILURE${NC}"

            # If we already found a success, we can stop
            if [ "$FOUND_SUCCESS" = true ]; then
                echo ""
                echo -e "${GREEN}Found both success and failure - stopping search${NC}"
                break
            fi
        else
            echo -e "${RED}Failure (already found)${NC}"
        fi
    else
        echo -e "${YELLOW}Conclusion: ${JOB_CONCLUSION}${NC}"
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RESULTS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$FOUND_SUCCESS" = true ]; then
    echo -e "${GREEN}✓ LAST SUCCESSFUL RUN:${NC}"
    echo -e "  Run: ${LAST_SUCCESSFUL_RUN}"
    echo -e "  Run ID: ${LAST_SUCCESSFUL_RUN_ID}"
    echo -e "  Commit: ${LAST_SUCCESSFUL_COMMIT}"
    echo -e "  Commit URL: ${BASE_URL}/commit/${LAST_SUCCESSFUL_COMMIT}"
    echo ""
else
    echo -e "${YELLOW}⚠ No successful run found in analyzed runs${NC}"
    echo ""
fi

if [ "$FOUND_FAILURE" = true ]; then
    echo -e "${RED}✗ FIRST FAILING RUN:${NC}"
    echo -e "  Run: ${FIRST_FAILING_RUN}"
    echo -e "  Run ID: ${FIRST_FAILING_RUN_ID}"
    echo -e "  Commit: ${FIRST_FAILING_COMMIT}"
    echo -e "  Commit URL: ${BASE_URL}/commit/${FIRST_FAILING_COMMIT}"
    echo ""
else
    echo -e "${YELLOW}⚠ No failing run found in analyzed runs${NC}"
    echo ""
fi

if [ "$FOUND_SUCCESS" = true ] && [ "$FOUND_FAILURE" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}COMMIT RANGE${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "Commits between successful and failing runs:"
    echo -e "  ${BASE_URL}/compare/${LAST_SUCCESSFUL_COMMIT}...${FIRST_FAILING_COMMIT}"
    echo ""

    # Try to get commit count
    COMMIT_COUNT=$(git rev-list --count "${LAST_SUCCESSFUL_COMMIT}..${FIRST_FAILING_COMMIT}" 2>/dev/null || echo "unknown")
    echo -e "  Commit count: ${COMMIT_COUNT}"
    echo ""
fi

if [ "$FOUND_SUCCESS" = false ] && [ "$FOUND_FAILURE" = false ]; then
    echo -e "${RED}Error: Could not find any runs with subjob '${SUBJOB_NAME}'${NC}"
    echo "Make sure the subjob name is correct and exists in the workflow."
    exit 1
fi
