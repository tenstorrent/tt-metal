# Add Self-Review TODO

## Overview
Add a self-review TODO item to the PR tracking file when you notice something that needs to be addressed during code review.

## Input
- File and line range with the issue
- Description of what needs to be fixed

## Steps

1. **Read the tracking file**
   - Open `_local/{PR_ID}_{name}.md`

2. **Add the self-review item**
   - Add before "## General PR Comments" section
   - Use format:
     ```
     📝 **Comment addressed status:** TODO
     Self-review
     File: {file_path}:{line_range}

     **Context:**
     ```cpp
     {relevant_code}
     ```

     **Comment:** {description of what needs to be fixed}

     ---
     ```

3. **Remove TODO markers from code**
   - If the issue was marked with `// TODO: NOW:` or similar, remove it from the code
   - The tracking file is now the source of truth

## Notes
- Self-review items follow the same workflow as reviewer comments
- Use lowercase for comment descriptions
- Be specific about what needs to be changed
