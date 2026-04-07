# Project Guidelines

## Environment Setup

Always activate the virtual environment first:

```bash
source python_env/bin/activate
```

## Building

Build the project (requires venv to be active):

```bash
./build_metal.sh
```

## Running Tests

Run tests with pytest:

```bash
python -m pytest <test_path> -v
```

## Git Push

SSH is not authorized for SAML SSO. Use HTTPS for pushing:

```bash
git remote set-url origin https://github.com/tenstorrent/tt-metal.git
git push origin <branch>
git remote set-url origin git@github.com:tenstorrent/tt-metal.git
```

Or run `gh auth setup-git` first to configure git to use GitHub CLI credentials.
