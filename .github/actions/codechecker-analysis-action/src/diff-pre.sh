#!/bin/bash
if [[ ! -z "$CODECHECKER_ACTION_DEBUG" ]]; then
  set -x
fi

if [[ -z "$IN_DIFF_URL" ]]; then
  echo "::error title=Configuration error::Diffing results against a server was enabled, but the product URL is not configured."
  exit 1
fi

if [[ ! -z "$IN_DIFF_USERNAME" && ! -z "$IN_DIFF_PASSWORD" ]]; then
  echo "Configuring credentials..."
  cat <<EOF > ~/.codechecker.passwords.json
    {
      "client_autologin": true,
      "credentials": {
        "$IN_DIFF_URL": "$IN_DIFF_USERNAME:$IN_DIFF_PASSWORD"
      }
    }
EOF
  chmod 0600 ~/.codechecker.passwords.json
fi

if [[ ! -z "$IN_DIFF_RUN_NAME" && "$IN_DIFF_RUN_NAME" != "__DEFAULT__" ]]; then
  echo "Using user-requested run name."
  echo "RUN_NAME=$IN_DIFF_RUN_NAME" >> "$GITHUB_OUTPUT"
  echo "DIFF_CONFIGURED=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

if [[ "$GITHUB_EVENT_NAME" == "pull_request" ]]; then
  echo "Auto-generating run name for a PULL REQUEST's target (base)."
  echo "RUN_NAME=$GITHUB_REPOSITORY\: $GITHUB_BASE_REF" >> "$GITHUB_OUTPUT"
  echo "DIFF_CONFIGURED=true" >> "$GITHUB_OUTPUT"
  exit 0
elif [[ "$GITHUB_REF_TYPE" == "branch" ]]; then
  echo "Auto-generating run name for a BRANCH."
  echo "RUN_NAME=$GITHUB_REPOSITORY\: $GITHUB_REF_NAME" >> "$GITHUB_OUTPUT"
  echo "DIFF_CONFIGURED=true" >> "$GITHUB_OUTPUT"
  exit 0
elif [[ "$GITHUB_REF_TYPE" == "tag" ]]; then
  echo "Auto-generating run name for a TAG."
  echo "RUN_NAME=$GITHUB_REPOSITORY tags" >> "$GITHUB_OUTPUT"
  echo "DIFF_CONFIGURED=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "::notice title=Preparation for diff::Failed to generate a run name. Implementation error?"
echo "DIFF_CONFIGURED=false" >> "$GITHUB_OUTPUT"
