GIT_HOOKS = $(OUT)/git_hooks

git_hooks: $(GIT_HOOKS)/.installed

$(GIT_HOOKS)/.installed: python_env/dev
	mkdir -p $(GIT_HOOKS)
	bash -c "source $(PYTHON_ENV)/bin/activate && pre-commit install"
	bash -c "source $(PYTHON_ENV)/bin/activate && pre-commit install --hook-type commit-msg"
	touch $@
