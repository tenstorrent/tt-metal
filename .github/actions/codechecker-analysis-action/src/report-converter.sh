#!/bin/bash
if [[ ! -z "$CODECHECKER_ACTION_DEBUG" ]]; then
  set -x
fi

echo "::group::Preparing for conversion"
if [[ -z "$IN_ORIGINAL_ANALYSER" ]]; then
  echo "::error title=Internal error::environment variable 'IN_ORIGINAL_ANALYSER' missing!"
  exit 1
fi
if [[ -z "$IN_ORIGINAL_ANALYSIS_OUTPUT" ]]; then
  echo "::error title=Internal error::environment variable 'IN_ORIGINAL_ANALYSIS_OUTPUT' missing!"
  exit 1
fi

OUTPUT_DIR="$IN_OUTPUT_DIR"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR=~/"$GITHUB_ACTION_NAME"_Results
fi

mkdir -pv "$(dirname $"OUTPUT_DIR")"

# Report-Converter does not support a config file. :(
# if [[ ! -z "$IN_CONFIGFILE" ]]; then
#   CONFIG_FLAG_1="--config"
#   CONFIG_FLAG_2=$IN_CONFIGFILE
#   echo "Using configuration file \"$IN_CONFIGFILE\"!"
# fi
echo "::endgroup::"

echo "::group::Performing conversion"
"$CODECHECKER_PATH"/report-converter \
    "$IN_ORIGINAL_ANALYSIS_OUTPUT" \
    --type "$IN_ORIGINAL_ANALYSER" \
    --output "$OUTPUT_DIR" \
    --export plist
EXIT_CODE=$?
echo "::endgroup::"

if [[ $EXIT_CODE -ne 0 && "$IN_IGNORE_CRASHES" == "true" ]]; then
  # In general it is a good idea not to destroy the entire job just because a
  # few translation units failed. Crashes are, unfortunately, usual.
  echo "::warning title=Report Converter crashed on some inputs::Some of the analysis results failed to convert due to internal error."
  EXIT_CODE=0
fi

echo "OUTPUT_DIR=$OUTPUT_DIR" >> "$GITHUB_OUTPUT"
exit $EXIT_CODE
