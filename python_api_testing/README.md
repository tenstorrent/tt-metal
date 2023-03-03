# Running the python tests
After you have compiled the project in project root, run the following to be able
to run the python tests
```
cd $BUDA_HOME # Make sure you exported this env var already
source build/python_env/bin/activate
pip install -r python_api_testing/requirements.txt
python python_api_testing/models/bert/mha.py # Example test
```
