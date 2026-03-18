# Git Commands

## Testing on a Separate Branch

To test changes without modifying your current branch, create a new branch:

```bash
git checkout -b my-test-branch
```

Any commits will stay on `my-test-branch`. When done, switch back:

```bash
git checkout <original-branch>
```

To discard the test branch afterward:

```bash
git branch -D my-test-branch
```
