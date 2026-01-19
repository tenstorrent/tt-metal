#!/bin/bash
if [[ ! -z "$CODECHECKER_ACTION_DEBUG" ]]; then
  set -x
fi

if [[ ! -z "$IN_LOGFILE" && ! -z "$IN_COMMAND" ]]; then
  echo "::error title=Configuration error::'logfile' and 'build-command' both specified!"
  exit 1
fi

mkdir -pv $(dirname "$OUT_FILE")

EXIT_CODE=0

if [[ ! -z "$IN_LOGFILE" ]]; then
  # Pretty trivial.
  cp -v "$IN_LOGFILE" "$OUT_FILE"
  EXIT_CODE=$?
elif [[ ! -z "$IN_COMMAND" ]]; then
  echo "::group::Creating a build log by executing the build"
  "$CODECHECKER_PATH"/CodeChecker log \
    --build "$IN_COMMAND" \
    --output "$OUT_FILE"
  EXIT_CODE=$?
  echo "::endgroup::"
else
  echo "::error title=Configuration error::neither 'logfile' nor 'build-command' specified!"
  echo "[]" > "$OUT_FILE"
  exit 1
fi

echo "COMPILATION_DATABASE=$OUT_FILE" >> "$GITHUB_OUTPUT"
exit $EXIT_CODE
