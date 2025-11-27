# Address PR Review Comment

## Overview
Address a specific PR review comment, make code changes, and update the tracking file.

## Input
- Reference to the comment in `_local/{PR_ID}_{name}.md` (line range or quoted text)

## Steps

1. **Understand the comment**
   - Read the full context from the tracking file
   - Read the relevant source files
   - Understand what change is requested

2. **Make the code change**
   - Apply the fix as requested by the reviewer
   - Follow project conventions:
     - Objects: use `const&` for parameters
     - Simple types (bool, int): no `const` for pass-by-value
     - Include order: local → project → std
     - No emojis in code comments

3. **Wait for confirmation before committing**
   - Show the proposed changes
   - Wait for user to accept/modify

4. **Update the tracking file**
   - Change status from `📝 TODO` to `⚠️ ADDRESSED, TO MERGE AND LEAVE COMMENT`
   - Update the context to show the fixed code
   - Add `**My comment:** {lowercase description of what was done}`

5. **Commit when approved**
   - Use `fix:` prefix for the commit message
   - Be concise and specific about what was changed
   - After committing, check if clang-format modified any files
   - If so, add those changes and amend the commit

## Notes
- Do NOT commit without user confirmation
- Pre-commit hooks (clang-format) may modify files - always check `git status` after commit
- If clang-format changes files, add them and amend or create a combined commit
- Keep commit messages focused on the specific fix
