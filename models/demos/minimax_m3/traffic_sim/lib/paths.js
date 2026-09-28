// Where data and results live. Order: environment override, the project's own ./data and ./results
// (standalone repository layout), then the exabox scratch location used by the tt-metal copy.
'use strict';
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..');
function pick(envName, local, exabox) {
  if (process.env[envName]) return path.resolve(process.env[envName]);
  if (fs.existsSync(local)) return local;
  if (fs.existsSync(exabox)) return exabox;
  return local;
}
const DATA = pick('M3SIM_DATA', path.join(ROOT, 'data'), '/data/philei/m3_traffic_sim/data');
const RESULTS = pick('M3SIM_RESULTS', path.join(ROOT, 'results'), '/data/philei/m3_traffic_sim/results');
module.exports = { ROOT, DATA, RESULTS, STUDY: path.join(RESULTS, 'study.json'), DIST: path.join(ROOT, 'dist') };
