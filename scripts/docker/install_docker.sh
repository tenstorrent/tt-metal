#!/bin/bash

# Setup sources for docker
sudo apt-get update -y
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y

# Install docker and plugins
sudo apt-get install -y \
  docker-ce=5:26.1.1-1~ubuntu.20.04~focal \
  docker-ce-cli=5:26.1.1-1~ubuntu.20.04~focal \
  containerd.io=1.6.31-1 \
  docker-buildx-plugin=0.14.0-1~ubuntu.20.04~focal \
  docker-compose-plugin=2.27.0-1~ubuntu.20.04~focal

# Set up docker for non-root user
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
