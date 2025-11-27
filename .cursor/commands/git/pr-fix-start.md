# Start PR Fix Session

## Overview
Start a PR fix session by creating a `.fix` branch and collecting all review comments to a local markdown file for tracking.

## Input
- PR URL (e.g., `https://github.com/tenstorrent/tt-metal/pull/33310`)

## Steps

1. **Create .fix branch**
   - Get current branch name (should be the PR branch)
   - Create a new branch with `.fix` suffix (e.g., `shutov/tmp_argmax_op.fix`)
   - Push to remote

2. **Ensure _local/ is in .gitignore**
   - Add `_local/` to `.gitignore` if not present
   - Commit with `LOCAL:` prefix (so it's clear this commit should NOT be cherry-picked)

3. **Collect PR comments using GitHub MCP**
   - Use `pull_request_read` with `get_review_comments` method
   - Use `pull_request_read` with `get_comments` method for general comments
   - Create `_local/{PR_ID}_{short_name}.md` file

4. **Format the comments file**
   - Add status flow legend at the top:
     ```
     **Status flow:** 📝 TODO → ⚠️ ADDRESSED, TO MERGE AND LEAVE COMMENT → ✅ DONE
     (📝 TODO: not started | ⚠️ ADDRESSED: fixed in .fix branch, needs cherry-pick and comment | ✅ DONE: cherry-picked and comment left)
     ```
   - For each review comment:
     ```
     📝 **Comment addressed status:** TODO
     {comment_html_url} @{reviewer_login}
     File: {file_path}:{line}

     **Context:**
     ```cpp
     {code_context}
     ```

     **Comment:** {comment_body}

     ---
     ```

5. **Add conventions reminder section** (before "## General PR Comments")
   ```markdown
   ## Conventions Reminder (check when editing files)

   ### Copyright Header
   - Use `© 2025 Tenstorrent AI ULC.` (not "Tenstorrent Inc.")
   - C++: `// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.`
   - Python: `# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.`
   - No empty line between copyright and license

   ### Include Sorting (C++)
   - Order: local → project → std
   - Remove unused includes

   ### Const Correctness (C++)
   - Objects: use `const&` (e.g., `const std::optional<int>& dim`)
   - Simple types: no `const` for pass-by-value (e.g., `bool keepdim`)

   ---
   ```

## Notes
- The `.fix` branch allows dirty work without triggering CI on every commit
- `LOCAL:` prefixed commits should NOT be cherry-picked to the PR branch
- Status flow: TODO → ADDRESSED, TO MERGE AND LEAVE COMMENT → DONE
- When a fix is done, update status to ⚠️ ADDRESSED and add `**My comment:** {description}`
