#!/bin/bash
if [[ ! -z "$CODECHECKER_ACTION_DEBUG" ]]; then
  set -x
fi

echo "::group::Preparing for store"

if [[ -z "$IN_STORE_URL" ]]; then
  echo "::error title=Internal error::environment variable 'IN_STORE_URL' missing!"
  exit 1
fi

if [[ -z "$PROJECT_PATH" ]]; then
  echo "::error title=Internal error::environment variable 'PROJECT_PATH' missing!"
  exit 1
fi

if [[ -z "$RAW_RESULT_DIR" ]]; then
  echo "::error title=Internal error::environment variable 'RAW_RESULT_DIR' missing!"
  exit 1
fi

if [[ -z "$CODECHECKER_STORE_RUN_NAME" ]]; then
  echo "::error title=Internal error::environment variable 'CODECHECKER_STORE_RUN_NAME' missing!"
  exit 1
fi

if [[ ! -z "$IN_CONFIGFILE" ]]; then
  CONFIG_FLAG_1="--config"
  CONFIG_FLAG_2=$IN_CONFIGFILE
  echo "Using configuration file \"$IN_CONFIGFILE\"!"
fi

if [[ ! -z "$CODECHECKER_STORE_RUN_TAG" ]]; then
  RUN_TAG_FLAG_1="--tag"
  RUN_TAG_FLAG_2=$CODECHECKER_STORE_RUN_TAG
fi
echo "::endgroup::"

echo "::group::Storing results to server"
"$CODECHECKER_PATH"/CodeChecker \
  store \
  "$RAW_RESULT_DIR" \
  --url "$IN_STORE_URL" \
  --name "$CODECHECKER_STORE_RUN_NAME" \
  --trim-path-prefix "$PROJECT_PATH" \
  $RUN_TAG_FLAG_1 $RUN_TAG_FLAG_2 \
  $CONFIG_FLAG_1 $CONFIG_FLAG_2
SUCCESS=$?
echo "::endgroup::"

if [[ $SUCCESS -ne 0 ]]; then
  echo "::warning title=Storing results failed::Executing 'CodeChecker store' to upload analysis results to the server has failed. The logs usually provide more information."
  echo "SUCCESS=false" >> "$GITHUB_OUTPUT"
else
  echo "SUCCESS=true" >> "$GITHUB_OUTPUT"
fi

# Always return 0 from this step. The user can decide if storage is mandatory
# and break the build later.
exit 0
