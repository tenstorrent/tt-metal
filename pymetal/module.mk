include pymetal/csrc/module.mk

pymetal: pymetal/csrc pymetal/csrc/setup_local_so
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu"
