# Installing Docs for Development

Docs will be available on localhost:<port>/index.html

```
cd docs/

# Create python env with sphinx 
python3 -m venv env
source env/bin/activate
pip install -r requirements-dev.txt

# Build the docs
source env/bin/activate
PORT=<port> make all
```
