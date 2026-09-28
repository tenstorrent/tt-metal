#!/usr/bin/env node
// Serve the lab as a local web app. No dependencies (Node >= 18).
//   node server.js [--port 8765] [--host 127.0.0.1] [--rebuild]
// Builds dist/index.html from the template, simulator, calibration, traffic and study results when it is missing
// or older than any of its inputs, then serves it. The simulation itself runs in the browser (Web Workers).
// From a laptop: ssh -L 8765:localhost:8765 <server>, then open http://localhost:8765
'use strict';
const http = require('http'), fs = require('fs'), path = require('path'), { execFileSync } = require('child_process');
const P = require('./lib/paths.js');

const args = process.argv.slice(2);
const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const port = Number(get('--port', process.env.PORT || 8765));
const host = get('--host', process.env.HOST || '127.0.0.1');
const page = path.join(P.DIST, 'index.html');

const inputs = ['artifact/template.html', 'sim_core.js', 'presets.js', 'feature_details.js', 'study.js', 'build_artifact.js', 'calib_data.json']
  .map((f) => path.join(__dirname, f)).concat([path.join(P.DATA, 'traffic.bin'), path.join(P.DATA, 'traffic.json'), P.STUDY]);
for (const f of inputs) if (!fs.existsSync(f)) { console.error(`missing input ${f} (see README: data/ and results/)`); process.exit(1); }
const mtime = (f) => fs.statSync(f).mtimeMs;
if (args.includes('--rebuild') || !fs.existsSync(page) || inputs.some((f) => mtime(f) > mtime(page))) {
  console.log('building', page);
  execFileSync(process.execPath, [path.join(__dirname, 'build_artifact.js'), '--standalone', '--out', page], { stdio: 'inherit' });
}

const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405); res.end(); return; }
  if (url === '/' || url === '/index.html') {
    const body = fs.readFileSync(page);
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': body.length, 'Cache-Control': 'no-cache' });
    res.end(req.method === 'HEAD' ? undefined : body);
    return;
  }
  if (url === '/healthz') { res.writeHead(200, { 'Content-Type': 'text/plain' }); res.end('ok\n'); return; }
  res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('not found\n');
});
server.listen(port, host, () => {
  console.log(`M3 AgentX Prefill Lab on http://${host}:${port}/`);
  if (host === '127.0.0.1' || host === 'localhost') console.log(`remote machine? forward it: ssh -L ${port}:localhost:${port} <this host>, then open http://localhost:${port}`);
});
