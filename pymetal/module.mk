include pymetal/csrc/module.mk

$(PYTHON_ENV)/lib/python3.8/site-packages/ttlib.egg-link: python_env pymetal/csrc/setup_local_so
	bash -c "source $(PYTHON_ENV)/bin/activate; cd pymetal; pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu"

pymetal: pymetal/csrc pymetal/csrc/setup_local_so # $(PYTHON_ENV)/lib/python3.8/site-packages/ttlib.egg-link;
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu"
