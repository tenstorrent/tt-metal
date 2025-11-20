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
DATA_DIR="auto_triage/data"
SUMMARY_JSON_PATH="${DATA_DIR}/boundaries_summary.json"
RUNS_JSON_PATH="${DATA_DIR}/subjob_runs.json"

mkdir -p "$DATA_DIR"
rm -f "$SUMMARY_JSON_PATH" "$RUNS_JSON_PATH"

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
echo "Processing workflow runs page by page (this may take some time)..."
PER_PAGE=100
PAGE=1
TOTAL_RUNS_FETCHED=0
VALID_RUNS_FETCHED=0

LAST_SUCCESSFUL_RUN=""
LAST_SUCCESSFUL_RUN_ID=""
LAST_SUCCESSFUL_COMMIT=""
LAST_SUCCESSFUL_JOB_URL=""
FIRST_FAILING_RUN=""
FIRST_FAILING_RUN_ID=""
FIRST_FAILING_COMMIT=""
FIRST_FAILING_JOB_URL=""

PROCESSED=0
FOUND_SUCCESS=false

# Track the most recent failure we see, then when we find the last success,
# that failure is the first failure after the success
MOST_RECENT_FAILURE_RUN=""
MOST_RECENT_FAILURE_RUN_ID=""
MOST_RECENT_FAILURE_COMMIT=""
MOST_RECENT_FAILURE_JOB_URL=""
FAILED_RUNS_JSON='[]'
SUBJOB_RUNS_JSON='[]'

while true; do
    PAGE_RESPONSE=$(gh api "repos/${REPO}/actions/workflows/${WORKFLOW_ID}/runs?branch=main&per_page=${PER_PAGE}&page=${PAGE}" 2>/dev/null || echo "")
    if [ -z "$PAGE_RESPONSE" ]; then
        if [ "$PAGE" -eq 1 ]; then
            echo -e "${RED}Error: Could not fetch workflow runs${NC}"
            exit 1
        fi
        break
    fi

    RUNS_PAGE=$(echo "$PAGE_RESPONSE" | jq '.workflow_runs // []')
    PAGE_TOTAL=$(echo "$RUNS_PAGE" | jq 'length')

    if [ "$PAGE_TOTAL" -eq 0 ]; then
        if [ "$PAGE" -eq 1 ]; then
            echo -e "${RED}Error: No workflow runs returned${NC}"
            exit 1
        fi
        break
    fi

    TOTAL_RUNS_FETCHED=$((TOTAL_RUNS_FETCHED + PAGE_TOTAL))
    echo -e "${GREEN}Fetched ${TOTAL_RUNS_FETCHED} workflow runs so far (page ${PAGE})${NC}"

    VALID_PAGE=$(echo "$RUNS_PAGE" | jq -r "[.[] | select(.head_branch == \"main\" and .status == \"completed\" and .conclusion != \"cancelled\")]")
    VALID_COUNT=$(echo "$VALID_PAGE" | jq 'length')

    if [ "$VALID_COUNT" -eq 0 ]; then
        PAGE=$((PAGE + 1))
        continue
    fi

    VALID_RUNS_FETCHED=$((VALID_RUNS_FETCHED + VALID_COUNT))
    echo -e "${GREEN}Valid runs accumulated: ${VALID_RUNS_FETCHED}${NC}"

    while IFS= read -r RUN_DATA; do
        RUN_ID=$(echo "$RUN_DATA" | jq -r '.id')
        RUN_COMMIT=$(echo "$RUN_DATA" | jq -r '.head_sha')
        RUN_COMPLETED_AT=$(echo "$RUN_DATA" | jq -r '.updated_at // .run_started_at // "unknown"')
        RUN_URL="${BASE_URL}/actions/runs/${RUN_ID}"

        PROCESSED=$((PROCESSED + 1))
        echo -n "[${PROCESSED}] Checking run ${RUN_ID} (${RUN_COMPLETED_AT})... "

        # Fetch jobs page by page, stopping once we find the subjob
        PAGE_J=1
        FOUND_JOB=false

        while [ "$FOUND_JOB" = false ]; do
            PAGE_JOBS=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}/jobs?per_page=${PER_PAGE}&page=${PAGE_J}" --jq '[.jobs[] | {name, conclusion, status, id}]' 2>/dev/null || echo "[]")

            if [ "$PAGE_JOBS" = "[]" ] || [ -z "$PAGE_JOBS" ]; then
                break
            fi

            JOB_COUNT=$(echo "$PAGE_JOBS" | jq 'length' 2>/dev/null || echo "0")
            if [ "$JOB_COUNT" -eq 0 ]; then
                break
            fi

            SUBJOB=$(echo "$PAGE_JOBS" | jq -r ".[] | select(.name == \"${SUBJOB_NAME}\" or .name == \"single-card-demo-tests / ${SUBJOB_NAME}\" or .name == \"${WORKFLOW_NAME} / ${SUBJOB_NAME}\" or (.name | endswith(\"${SUBJOB_NAME}\")) or (.name | contains(\"${SUBJOB_NAME}\")))" || echo "")

            if [ -n "$SUBJOB" ]; then
                FOUND_JOB=true
                break
            fi

            if [ "$JOB_COUNT" -lt "$PER_PAGE" ]; then
                break
            fi

            PAGE_J=$((PAGE_J + 1))
        done

        if [ "$FOUND_JOB" = false ] || [ -z "$SUBJOB" ]; then
            echo -e "${YELLOW}Subjob not found${NC}"
            continue
        fi

        JOB_CONCLUSION=$(echo "$SUBJOB" | jq -r '.conclusion // "null"')
        JOB_STATUS=$(echo "$SUBJOB" | jq -r '.status')
        JOB_ID=$(echo "$SUBJOB" | jq -r '.id')
        JOB_URL="${BASE_URL}/actions/runs/${RUN_ID}/job/${JOB_ID}"

        if [ "$JOB_STATUS" != "completed" ]; then
            echo -e "${YELLOW}Job not completed${NC}"
            continue
        fi

        if [ "$JOB_CONCLUSION" = "success" ]; then
            if [ "$FOUND_SUCCESS" = false ]; then
                LAST_SUCCESSFUL_RUN="$RUN_URL"
                LAST_SUCCESSFUL_RUN_ID="$RUN_ID"
                LAST_SUCCESSFUL_COMMIT="$RUN_COMMIT"
                LAST_SUCCESSFUL_JOB_URL="$JOB_URL"
                FOUND_SUCCESS=true
                SUBJOB_RUNS_JSON=$(jq -n \
                    --arg status "success" \
                    --arg run_url "$RUN_URL" \
                    --arg job_url "$JOB_URL" \
                    --arg run_id "$RUN_ID" \
                    --arg job_id "$JOB_ID" \
                    --arg commit "$RUN_COMMIT" \
                    --arg completed_at "$RUN_COMPLETED_AT" \
                    --argjson arr "$SUBJOB_RUNS_JSON" \
                    --argjson run_number "$PROCESSED" \
                    '$arr + [{status:$status, run_url:$run_url, job_url:$job_url, run_id:$run_id, job_id:$job_id, commit:$commit, completed_at:$completed_at, run_number:$run_number}]' \
                )

                if [ -n "$MOST_RECENT_FAILURE_RUN" ]; then
                    FIRST_FAILING_RUN="$MOST_RECENT_FAILURE_RUN"
                    FIRST_FAILING_RUN_ID="$MOST_RECENT_FAILURE_RUN_ID"
                    FIRST_FAILING_COMMIT="$MOST_RECENT_FAILURE_COMMIT"
                    FIRST_FAILING_JOB_URL="$MOST_RECENT_FAILURE_JOB_URL"
                fi

                echo -e "${GREEN}✓ SUCCESS (last successful)${NC}"
                echo ""
                echo -e "${GREEN}Found last success and first failure - stopping search${NC}"
                break
            fi
        elif [ "$JOB_CONCLUSION" = "failure" ]; then
            MOST_RECENT_FAILURE_RUN="$RUN_URL"
            MOST_RECENT_FAILURE_RUN_ID="$RUN_ID"
            MOST_RECENT_FAILURE_COMMIT="$RUN_COMMIT"
            MOST_RECENT_FAILURE_JOB_URL="$JOB_URL"
            echo -e "${RED}✗ FAILURE${NC}"
            FAILED_RUNS_JSON=$(jq -n \
                --arg run_url "$RUN_URL" \
                --arg job_url "$JOB_URL" \
                --arg run_id "$RUN_ID" \
                --arg job_id "$JOB_ID" \
                --arg commit "$RUN_COMMIT" \
                --arg completed_at "$RUN_COMPLETED_AT" \
                --arg conclusion "$JOB_CONCLUSION" \
                --argjson arr "$FAILED_RUNS_JSON" \
                --argjson run_number "$PROCESSED" \
                '$arr + [{run_url:$run_url, job_url:$job_url, run_id:$run_id, job_id:$job_id, commit:$commit, completed_at:$completed_at, conclusion:$conclusion, run_number:$run_number}]' \
            )
            SUBJOB_RUNS_JSON=$(jq -n \
                --arg status "failure" \
                --arg run_url "$RUN_URL" \
                --arg job_url "$JOB_URL" \
                --arg run_id "$RUN_ID" \
                --arg job_id "$JOB_ID" \
                --arg commit "$RUN_COMMIT" \
                --arg completed_at "$RUN_COMPLETED_AT" \
                --argjson arr "$SUBJOB_RUNS_JSON" \
                --argjson run_number "$PROCESSED" \
                '$arr + [{status:$status, run_url:$run_url, job_url:$job_url, run_id:$run_id, job_id:$job_id, commit:$commit, completed_at:$completed_at, run_number:$run_number}]' \
            )
        else
            echo -e "${YELLOW}Conclusion: ${JOB_CONCLUSION}${NC}"
        fi
    done < <(echo "$VALID_PAGE" | jq -c '.[]')

    if [ "$FOUND_SUCCESS" = true ]; then
        break
    fi

    PAGE=$((PAGE + 1))
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RESULTS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

FOUND_FAILURE=false
if [ -n "$FIRST_FAILING_RUN" ]; then
    FOUND_FAILURE=true
fi

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

COMPARE_URL=""
COMMIT_COUNT=""

if [ "$FOUND_SUCCESS" = true ] && [ "$FOUND_FAILURE" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}COMMIT RANGE${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "Commits between successful and failing runs:"
    COMPARE_URL="${BASE_URL}/compare/${LAST_SUCCESSFUL_COMMIT}...${FIRST_FAILING_COMMIT}"
    echo -e "  ${COMPARE_URL}"
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

if [ "$SUBJOB_RUNS_JSON" = "[]" ]; then
    echo -e "${BLUE}No qualifying subjob runs recorded.${NC}"
else
    echo -e "${BLUE}Recorded subjob runs (success + failure):${NC}"
    echo "$SUBJOB_RUNS_JSON"
fi

if [ "$FAILED_RUNS_JSON" != "[]" ]; then
    echo -e "${BLUE}Failed subjobs (JSON subset):${NC}"
    echo "$FAILED_RUNS_JSON"
fi

if [ -n "$SUMMARY_JSON_PATH" ]; then
    tmp_summary="$(mktemp)"
    jq -n \
        --arg workflow "$WORKFLOW_NAME" \
        --arg subjob "$SUBJOB_NAME" \
        --arg last_success_commit "${LAST_SUCCESSFUL_COMMIT:-}" \
        --arg last_success_run "${LAST_SUCCESSFUL_RUN:-}" \
        --arg last_success_run_id "${LAST_SUCCESSFUL_RUN_ID:-}" \
        --arg last_success_job "${LAST_SUCCESSFUL_JOB_URL:-}" \
        --arg first_fail_commit "${FIRST_FAILING_COMMIT:-}" \
        --arg first_fail_run "${FIRST_FAILING_RUN:-}" \
        --arg first_fail_run_id "${FIRST_FAILING_RUN_ID:-}" \
        --arg first_fail_job "${FIRST_FAILING_JOB_URL:-}" \
        --arg compare_url "${COMPARE_URL:-}" \
        --arg commit_count "${COMMIT_COUNT:-}" \
        --argjson runs "$SUBJOB_RUNS_JSON" \
        --argjson failed_runs "$FAILED_RUNS_JSON" \
        '{
            workflow: $workflow,
            subjob: $subjob,
            last_success: {
                commit: $last_success_commit,
                run_url: $last_success_run,
                run_id: $last_success_run_id,
                job_url: $last_success_job
            },
            first_failure: {
                commit: $first_fail_commit,
                run_url: $first_fail_run,
                run_id: $first_fail_run_id,
                job_url: $first_fail_job
            },
            compare_url: $compare_url,
            commit_count: (
                if ($commit_count | test("^[0-9]+$")) then ($commit_count | tonumber)
                elif ($commit_count | length) > 0 then $commit_count
                else null
                end
            ),
            runs: $runs,
            failed_runs: $failed_runs
        }' > "$tmp_summary"
    mv "$tmp_summary" "$SUMMARY_JSON_PATH"
    echo "$SUBJOB_RUNS_JSON" > "$RUNS_JSON_PATH"
fi
