#!/bin/bash

# First step is to cut down the number of test reports
# used
# $ find . -type f -exec ls -l {} + | sort -k 5 -n | awk '{print $9}'
# within artifacts/ dir to find which files were biggest, and representatively
# chose a couple of reports

# Ensure the target directory exists
mkdir -p modified

# Iterate over all .log files in the current directory
for logfile in *.log; do
    # Define the new filename and path
    newfile="modified/$logfile"

    # Create or overwrite the new file
    > "$newfile"

    # Get the total number of lines in the log file
    total_lines=$(wc -l < "$logfile")

    # Save the first 100 lines to the new file
    head -n 100 "$logfile" >> "$newfile"

    # Iterate through each line to find "UPLOAD-ARTIFACT-UUID" and save 50 lines before and after
    grep -n "UPLOAD-ARTIFACT-UUID" "$logfile" | while IFS=: read -r line_number line_content; do
        # Calculate the start and end line numbers for the 50 lines before and after
        start=$((line_number > 50 ? line_number - 50 : 1))
        end=$((line_number + 50 > total_lines ? total_lines : line_number + 50))

        # Save this chunk of lines to the new file
        sed -n "${start},${end}p" "$logfile" >> "$newfile"
    done

    # Save the last 100 lines to the new file
    tail -n 100 "$logfile" >> "$newfile"

    echo "[Info] Processed $logfile and saved to $newfile"
done
