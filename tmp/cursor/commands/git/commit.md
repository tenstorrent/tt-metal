# Commit

## Overview
Create a git commit with an appropriate commit message following conventional commit format.

## Steps
1. **Prepare changes**
   - Check what stages you got with `git status`
   - Stage proper changes with `git add`

2. **Write commit message**
   - Follow conventional commit format: `<type>(<scope>): <subject>`
   - Use imperative, present tense for subject
   - Include body if explanation needed
   - Reference issues in footer if applicable

3. **Address any issues surfaced by a precommit hook**
   - If clang format caused some files to change - re-add and commit again


**Commit types**
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation changes
   - `refactor`: Code refactoring
   - `test`: Adding or updating tests
   - `chore`: Maintenance tasks

**Common scopes**
   - `ttnn`, `tt_metal`, `model`

## Commit Template
- [ ] Changes are staged
- [ ] Commit message follows conventional format
- [ ] No sensitive data in changes
